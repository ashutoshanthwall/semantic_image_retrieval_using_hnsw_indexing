#include "embedding/text_encoder_process.hpp"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "data/errors.hpp"

extern char** environ;

namespace sir {

UniqueFd& UniqueFd::operator=(UniqueFd&& o) noexcept {
    if (this != &o) {
        reset();
        fd_ = o.release();
    }
    return *this;
}
UniqueFd::~UniqueFd() { reset(); }
void UniqueFd::reset() noexcept {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

namespace {
constexpr std::uint32_t kErrorBit = 0x80000000u;

struct SpawnActions {  // RAII for posix_spawn_file_actions_t
    posix_spawn_file_actions_t a;
    SpawnActions() { posix_spawn_file_actions_init(&a); }
    ~SpawnActions() { posix_spawn_file_actions_destroy(&a); }
};
}  // namespace

TextEncoderProcess::TextEncoderProcess(const std::filesystem::path& python,
                                       const std::filesystem::path& script,
                                       const std::filesystem::path& config,
                                       std::size_t expectedDim,
                                       const std::filesystem::path& logFile)
    : dim_(expectedDim) {
    if (!std::filesystem::exists(python))
        throw EmbeddingError("text encoder: Python interpreter not found: " + python.string());
    if (!std::filesystem::exists(script))
        throw EmbeddingError("text encoder: helper script not found: " + script.string());
    int sv[2];
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) != 0)
        throw EmbeddingError(std::string("text encoder: socketpair failed: ") + std::strerror(errno));
    UniqueFd parent(sv[0]), child(sv[1]);

    SpawnActions fa;  // dup2 clears CLOEXEC on the targets, so only fds 0/1/2 survive exec
    posix_spawn_file_actions_adddup2(&fa.a, child.get(), STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&fa.a, child.get(), STDOUT_FILENO);
    posix_spawn_file_actions_addopen(&fa.a, STDERR_FILENO, logFile.c_str(),
                                     O_WRONLY | O_CREAT | O_APPEND, 0644);
    const std::string py = python.string(), sc = script.string(), cf = config.string();
    Vector<char*> argv{const_cast<char*>(py.c_str()), const_cast<char*>(sc.c_str()),
                            const_cast<char*>(cf.c_str()), nullptr};
    const int rc = ::posix_spawn(&pid_, py.c_str(), &fa.a, nullptr, argv.data(), environ);
    if (rc != 0)
        throw EmbeddingError(std::string("text encoder: posix_spawn failed: ") + std::strerror(rc));
    child.reset();
    sock_ = std::move(parent);

    std::uint32_t hello = 0;
    try {
        recvAll(&hello, sizeof hello);  // server sends its dimension once the model is loaded
    } catch (const EmbeddingError&) {
        throw EmbeddingError("text encoder failed to start (see " + logFile.string() +
                             "; pass --encoder-log <file> to capture its output)");
    }
    if (hello != dim_)
        throw EmbeddingError("text encoder dimension " + std::to_string(hello) +
                             " != config dimension " + std::to_string(dim_));
}

TextEncoderProcess::~TextEncoderProcess() {
    if (pid_ <= 0) return;
    if (sock_.get() >= 0) {
        const std::uint32_t bye = 0;
        ::send(sock_.get(), &bye, sizeof bye, MSG_NOSIGNAL);
        ::shutdown(sock_.get(), SHUT_WR);
    }
    using namespace std::chrono_literals;
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    int status = 0;
    bool signalled = false;
    while (::waitpid(pid_, &status, WNOHANG) == 0) {
        if (std::chrono::steady_clock::now() > deadline) {
            ::kill(pid_, signalled ? SIGKILL : SIGTERM);
            if (signalled) {
                ::waitpid(pid_, &status, 0);
                break;
            }
            signalled = true;
        }
        std::this_thread::sleep_for(10ms);
    }
}

void TextEncoderProcess::sendAll(const void* data, std::size_t n) {
    const auto* p = static_cast<const char*>(data);
    while (n > 0) {
        const ssize_t w = ::send(sock_.get(), p, n, MSG_NOSIGNAL);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) throw EmbeddingError("text encoder: helper process is not accepting input");
        p += w;
        n -= static_cast<std::size_t>(w);
    }
}

void TextEncoderProcess::recvAll(void* data, std::size_t n) {
    auto* p = static_cast<char*>(data);
    while (n > 0) {
        const ssize_t r = ::recv(sock_.get(), p, n, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) throw EmbeddingError("text encoder: helper process exited unexpectedly");
        p += r;
        n -= static_cast<std::size_t>(r);
    }
}

Vector<float> TextEncoderProcess::encode(std::string_view text) {
    if (text.empty()) throw QueryError("query is empty");
    const auto len = static_cast<std::uint32_t>(text.size());
    sendAll(&len, sizeof len);
    sendAll(text.data(), text.size());
    std::uint32_t header = 0;
    recvAll(&header, sizeof header);
    if (header & kErrorBit) {
        std::string msg(header & ~kErrorBit, '\0');
        recvAll(msg.data(), msg.size());
        throw EmbeddingError("text encoder error: " + msg);
    }
    if (header != dim_ * sizeof(float))
        throw EmbeddingError("text encoder returned " + std::to_string(header) +
                             " bytes, expected " + std::to_string(dim_ * sizeof(float)));
    Vector<float> v(dim_);
    recvAll(v.data(), header);
    return v;
}

}  // namespace sir
