#pragma once
// RAII owner of the persistent Python CLIP text-encoder helper
// (scripts/text_encoder_server.py). The child's stdin/stdout are one end of a Unix
// socketpair; the protocol is documented in the script. Destruction asks the child to
// exit, waits, and escalates to SIGTERM/SIGKILL if it does not.
#include <filesystem>
#include <string_view>
#include <vector>

#include <sys/types.h>

#include "embedding/query_encoder.hpp"

namespace sir {

class UniqueFd {
public:
    UniqueFd() = default;
    explicit UniqueFd(int fd) noexcept : fd_(fd) {}
    UniqueFd(UniqueFd&& o) noexcept : fd_(o.release()) {}
    UniqueFd& operator=(UniqueFd&& o) noexcept;
    UniqueFd(const UniqueFd&) = delete;
    UniqueFd& operator=(const UniqueFd&) = delete;
    ~UniqueFd();
    int get() const noexcept { return fd_; }
    int release() noexcept { const int f = fd_; fd_ = -1; return f; }
    void reset() noexcept;

private:
    int fd_ = -1;
};

class TextEncoderProcess final : public QueryEncoder {
public:
    TextEncoderProcess(const std::filesystem::path& python, const std::filesystem::path& script,
                       const std::filesystem::path& config, std::size_t expectedDim,
                       const std::filesystem::path& logFile = "/dev/null");
    ~TextEncoderProcess() override;
    TextEncoderProcess(const TextEncoderProcess&) = delete;
    TextEncoderProcess& operator=(const TextEncoderProcess&) = delete;

    std::vector<float> encode(std::string_view text) override;

private:
    void sendAll(const void* data, std::size_t n);
    void recvAll(void* data, std::size_t n);

    pid_t pid_ = -1;
    UniqueFd sock_;
    std::size_t dim_;
};

}  // namespace sir
