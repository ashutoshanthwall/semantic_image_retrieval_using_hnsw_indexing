#include "persistence/index_manager.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <queue>
#include <set>

#include "data/errors.hpp"
#include "persistence/sha256.hpp"

namespace sir {
namespace {

constexpr char kMagic[8] = {'S', 'I', 'R', 'H', 'N', 'S', 'W', '\0'};

class Writer {
public:
    template <class T>
    void put(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* p = reinterpret_cast<const char*>(&v);
        buf_.insert(buf_.end(), p, p + sizeof(T));
    }
    void bytes(const void* p, std::size_t n) {
        const auto* c = static_cast<const char*>(p);
        buf_.insert(buf_.end(), c, c + n);
    }
    void str(const std::string& s) {
        put(static_cast<std::uint32_t>(s.size()));
        bytes(s.data(), s.size());
    }
    std::vector<char>& data() { return buf_; }

private:
    std::vector<char> buf_;
};

class Reader {
public:
    Reader(const std::vector<char>& buf, std::size_t end, std::string file)
        : buf_(buf), end_(end), file_(std::move(file)) {}
    template <class T>
    T get() {
        T v;
        need(sizeof(T));
        std::memcpy(&v, buf_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string str(std::size_t maxLen) {
        const auto n = get<std::uint32_t>();
        if (n > maxLen) throw PersistenceError(file_ + ": corrupted header (string length)");
        need(n);
        std::string s(buf_.data() + pos_, n);
        pos_ += n;
        return s;
    }
    std::string fixed(std::size_t n) {
        need(n);
        std::string s(buf_.data() + pos_, n);
        pos_ += n;
        return s;
    }
    std::size_t pos() const { return pos_; }

private:
    void need(std::size_t n) const {
        if (pos_ + n > end_)
            throw PersistenceError(file_ + ": index file is truncated (needed " +
                                   std::to_string(pos_ + n) + " bytes, have " +
                                   std::to_string(end_) + ")");
    }
    const std::vector<char>& buf_;
    std::size_t pos_ = 0;
    std::size_t end_;
    std::string file_;
};

}  // namespace

HNSWIndex IndexManager::build(const EmbeddingStore& store, const HnswParams& params,
                              BuildStats* stats) {
    const auto t0 = std::chrono::steady_clock::now();
    HNSWIndex index(store, params);
    for (std::size_t i = 0; i < store.size(); ++i) index.insert(static_cast<int>(i), store.row(i));
    if (stats)
        stats->seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return index;
}

void IndexManager::save(const HNSWIndex& index, const std::string& modelName,
                        const std::filesystem::path& path) {
    const EmbeddingStore& store = index.store();
    if (store.sourceSha256().size() != 64)
        throw PersistenceError("cannot save: the embedding store has no source SHA-256");
    Writer w;
    w.bytes(kMagic, sizeof(kMagic));
    w.put(kFormatVersion);
    w.str(modelName);
    w.put(static_cast<std::uint64_t>(store.dim()));
    w.put(static_cast<std::uint64_t>(store.size()));
    w.put(static_cast<std::uint64_t>(index.size()));
    const HnswParams& p = index.params();
    w.put(static_cast<std::int32_t>(p.M));
    w.put(static_cast<std::int32_t>(p.efConstruction));
    w.put(static_cast<std::int32_t>(p.efSearch));
    w.put(static_cast<std::uint64_t>(p.seed));
    w.put(static_cast<std::uint8_t>(p.selectHeuristic ? 1 : 0));
    w.put(static_cast<std::uint8_t>(p.keepPrunedConnections ? 1 : 0));
    w.put(static_cast<std::uint8_t>(p.protectLastInEdge ? 1 : 0));
    w.bytes(store.sourceSha256().data(), 64);
    w.put(static_cast<std::int32_t>(index.entryPoint()));
    w.put(static_cast<std::int32_t>(index.maxLevel()));
    w.str(index.rngState());
    Writer payload;
    for (std::size_t i = 0; i < store.size(); ++i) {
        const int id = static_cast<int>(i);
        const int level = index.levelOf(id);
        payload.put(static_cast<std::int32_t>(level));
        for (int l = 0; l <= level; ++l) {
            const auto nb = index.neighbors(id, l);
            payload.put(static_cast<std::uint32_t>(nb.size()));
            payload.bytes(nb.data(), nb.size() * sizeof(int));
        }
    }
    w.put(static_cast<std::uint64_t>(payload.data().size()));
    w.bytes(payload.data().data(), payload.data().size());
    Sha256 h;
    h.update(w.data().data(), w.data().size());
    const auto digest = h.finish();
    w.bytes(digest.data(), digest.size());

    std::filesystem::create_directories(path.parent_path());
    const auto tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out.write(w.data().data(), static_cast<std::streamsize>(w.data().size())))
            throw PersistenceError("failed writing " + tmp);
    }
    std::filesystem::rename(tmp, path);  // atomic replace: never a half-written index
}

HNSWIndex IndexManager::load(const std::filesystem::path& path, const EmbeddingStore& store,
                             const std::string& modelName) {
    const std::string f = path.string();
    if (!std::filesystem::exists(path))
        throw PersistenceError("index file not found: " + f +
                               " (build it with sir_build_index)");
    std::vector<char> buf(std::filesystem::file_size(path));
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.read(buf.data(), static_cast<std::streamsize>(buf.size())))
            throw PersistenceError("failed reading " + f);
    }
    if (buf.size() < sizeof(kMagic) + 4 + 32)
        throw PersistenceError(f + ": index file is truncated (" + std::to_string(buf.size()) +
                               " bytes)");
    if (std::memcmp(buf.data(), kMagic, sizeof(kMagic)) != 0)
        throw PersistenceError(f + ": not an index file (bad magic number)");

    const std::size_t body = buf.size() - 32;
    Reader r(buf, body, f);
    r.fixed(sizeof(kMagic));
    const auto version = r.get<std::uint32_t>();
    if (version != kFormatVersion)
        throw PersistenceError(f + ": unsupported format version " + std::to_string(version) +
                               " (expected " + std::to_string(kFormatVersion) + ")");
    const std::string model = r.str(256);
    const auto dim = r.get<std::uint64_t>();
    const auto rows = r.get<std::uint64_t>();
    const auto vectors = r.get<std::uint64_t>();
    HnswParams p;
    p.M = r.get<std::int32_t>();
    p.efConstruction = r.get<std::int32_t>();
    p.efSearch = r.get<std::int32_t>();
    p.seed = r.get<std::uint64_t>();
    p.selectHeuristic = r.get<std::uint8_t>() != 0;
    p.keepPrunedConnections = r.get<std::uint8_t>() != 0;
    p.protectLastInEdge = r.get<std::uint8_t>() != 0;
    const std::string sourceSha = r.fixed(64);
    const auto entry = r.get<std::int32_t>();
    const auto maxLevel = r.get<std::int32_t>();
    const std::string rng = r.str(1 << 16);
    const auto payloadBytes = r.get<std::uint64_t>();
    if (r.pos() + payloadBytes != body)
        throw PersistenceError(f + ": index file is truncated or has trailing data (header "
                               "declares " + std::to_string(r.pos() + payloadBytes + 32) +
                               " bytes, file has " + std::to_string(buf.size()) + ")");

    Sha256 h;
    h.update(buf.data(), body);
    const auto digest = h.finish();
    if (std::memcmp(digest.data(), buf.data() + body, 32) != 0)
        throw PersistenceError(f + ": index file is corrupted (SHA-256 checksum mismatch)");

    if (model != modelName)
        throw PersistenceError(f + ": stale index: built for model '" + model +
                               "', current model is '" + modelName + "' — rebuild it");
    if (dim != store.dim())
        throw PersistenceError(f + ": dimension mismatch: index dim " + std::to_string(dim) +
                               ", embeddings dim " + std::to_string(store.dim()));
    if (rows != store.size())
        throw PersistenceError(f + ": stale index: built on " + std::to_string(rows) +
                               " embeddings, current file has " + std::to_string(store.size()));
    if (sourceSha != store.sourceSha256())
        throw PersistenceError(f + ": stale index: built from embeddings " +
                               sourceSha.substr(0, 16) + "..., current embeddings are " +
                               store.sourceSha256().substr(0, 16) + "... — rebuild it");

    std::vector<HNSWIndex::NodeData> nodes(rows);
    const int maxDeg0 = 2 * p.M;
    for (std::size_t i = 0; i < rows; ++i) {
        auto& n = nodes[i];
        n.level = r.get<std::int32_t>();
        if (n.level < -1 || n.level > 64)
            throw PersistenceError(f + ": corrupted node level at id " + std::to_string(i));
        n.links.resize(static_cast<std::size_t>(n.level + 1));
        for (int l = 0; l <= n.level; ++l) {
            const auto cnt = r.get<std::uint32_t>();
            if (cnt > static_cast<std::uint32_t>(l == 0 ? maxDeg0 : p.M))
                throw PersistenceError(f + ": corrupted degree at id " + std::to_string(i));
            auto& links = n.links[static_cast<std::size_t>(l)];
            links.reserve(static_cast<std::size_t>(l == 0 ? maxDeg0 : p.M) + 1);
            for (std::uint32_t c = 0; c < cnt; ++c) {
                const auto e = r.get<std::int32_t>();
                if (e < 0 || static_cast<std::uint64_t>(e) >= rows)
                    throw PersistenceError(f + ": corrupted link at id " + std::to_string(i));
                links.push_back(e);
            }
        }
    }
    HNSWIndex index = HNSWIndex::restore(store, p, std::move(nodes), entry, maxLevel, rng);
    if (index.size() != vectors)
        throw PersistenceError(f + ": node count does not match header");
    validate(index);
    return index;
}

IndexValidation IndexManager::validate(const HNSWIndex& index) {
    IndexValidation v;
    const std::size_t n = index.store().size();
    v.nodes = index.size();
    v.maxLevel = index.maxLevel();
    if (v.nodes == 0) return v;
    if (!index.contains(index.entryPoint()) || index.levelOf(index.entryPoint()) != v.maxLevel)
        throw IndexError("invalid entry point " + std::to_string(index.entryPoint()));
    v.nodesPerLevel.assign(static_cast<std::size_t>(v.maxLevel) + 1, 0);
    std::size_t degSum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int id = static_cast<int>(i);
        if (!index.contains(id)) continue;
        const int level = index.levelOf(id);
        if (level > v.maxLevel)
            throw IndexError("node " + std::to_string(id) + " above the max level");
        for (int l = 0; l <= level; ++l) {
            ++v.nodesPerLevel[static_cast<std::size_t>(l)];
            const auto nb = index.neighbors(id, l);
            if (nb.size() > static_cast<std::size_t>(index.maxDegree(l)))
                throw IndexError("node " + std::to_string(id) + " exceeds max degree on layer " +
                                 std::to_string(l));
            std::set<int> uniq;
            for (int e : nb) {
                if (e == id) throw IndexError("self loop at node " + std::to_string(id));
                if (index.levelOf(e) < l)
                    throw IndexError("node " + std::to_string(id) + " links to " +
                                     std::to_string(e) + " which is absent from layer " +
                                     std::to_string(l));
                if (!uniq.insert(e).second)
                    throw IndexError("duplicate link at node " + std::to_string(id));
            }
            if (l == 0) degSum += nb.size();
        }
    }
    v.meanDegreeLayer0 = static_cast<double>(degSum) / static_cast<double>(v.nodes);
    std::vector<bool> seen(n, false);
    std::queue<int> q;
    q.push(index.entryPoint());
    seen[static_cast<std::size_t>(index.entryPoint())] = true;
    v.reachableLayer0 = 1;
    while (!q.empty()) {
        const int c = q.front();
        q.pop();
        for (int e : index.neighbors(c, 0))
            if (!seen[static_cast<std::size_t>(e)]) {
                seen[static_cast<std::size_t>(e)] = true;
                ++v.reachableLayer0;
                q.push(e);
            }
    }
    std::vector<std::uint32_t> inDeg(n, 0);
    for (std::size_t i = 0; i < n; ++i)
        if (index.contains(static_cast<int>(i)))
            for (int e : index.neighbors(static_cast<int>(i), 0)) ++inDeg[static_cast<std::size_t>(e)];
    for (std::size_t i = 0; i < n; ++i) {
        if (!index.contains(static_cast<int>(i))) continue;
        if (!seen[i]) v.unreachableLayer0.push_back(static_cast<int>(i));
        if (inDeg[i] == 0 && static_cast<int>(i) != index.entryPoint()) ++v.inDegreeZeroLayer0;
    }
    return v;
}

HNSWIndex IndexManager::rebuild(const EmbeddingStore& store, const HnswParams& params,
                                const std::string& modelName, const std::filesystem::path& path,
                                BuildStats* stats) {
    HNSWIndex index = build(store, params, stats);
    validate(index);
    save(index, modelName, path);
    return index;
}

}  // namespace sir
