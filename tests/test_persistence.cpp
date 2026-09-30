#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "data/errors.hpp"
#include "persistence/index_manager.hpp"
#include "persistence/sha256.hpp"
#include "test_helpers.hpp"

using namespace sir;
namespace fs = std::filesystem;

namespace {

EmbeddingStore makeStore(std::size_t n, std::size_t d, std::uint64_t seed) {
    auto v = test::randomUnitVectors(n, d, seed);
    const std::string sha = sha256Hex(v.data(), v.size() * sizeof(float));
    return EmbeddingStore(d, std::move(v), 1e-5, sha);
}

fs::path tmpFile(const std::string& name) {
    const auto dir = fs::temp_directory_path() / "sir_tests";
    fs::create_directories(dir);
    return dir / name;
}

HnswParams smallParams() {
    HnswParams p;
    p.M = 8;
    p.efConstruction = 64;
    p.efSearch = 40;
    p.seed = 7;
    return p;
}

void flipByte(const fs::path& p, std::size_t offset) {
    std::fstream f(p, std::ios::in | std::ios::out | std::ios::binary);
    f.seekg(static_cast<std::streamoff>(offset));
    char c;
    f.read(&c, 1);
    c = static_cast<char>(c ^ 0x5A);
    f.seekp(static_cast<std::streamoff>(offset));
    f.write(&c, 1);
}

std::string loadError(const fs::path& p, const EmbeddingStore& s, const std::string& model) {
    try {
        IndexManager::load(p, s, model);
    } catch (const PersistenceError& e) {
        return e.what();
    }
    return "";
}

}  // namespace

TEST(Persistence, RoundTripIsIdentical) {
    const auto store = makeStore(1500, 16, 1);
    const auto idx = IndexManager::build(store, smallParams());
    const auto path = tmpFile("roundtrip.idx");
    IndexManager::save(idx, "m", path);
    const auto loaded = IndexManager::load(path, store, "m");
    const auto qs = test::randomUnitVectors(50, 16, 2);
    for (int t = 0; t < 50; ++t) {
        std::span<const float> q(qs.data() + t * 16, 16);
        const auto a = idx.search(q, 10), b = loaded.search(q, 10);
        for (std::size_t i = 0; i < 10; ++i) {
            ASSERT_EQ(a[i].id, b[i].id);
            ASSERT_EQ(a[i].score, b[i].score);
        }
    }
    EXPECT_EQ(idx.rngState(), loaded.rngState());
    const auto v = IndexManager::validate(loaded);
    EXPECT_EQ(v.reachableLayer0, 1500u);
}

TEST(Persistence, RejectsCorruptTruncatedStaleAndMissing) {
    const auto store = makeStore(600, 8, 3);
    const auto idx = IndexManager::build(store, smallParams());
    const auto path = tmpFile("bad.idx");

    IndexManager::save(idx, "m", path);
    flipByte(path, fs::file_size(path) / 2);
    EXPECT_NE(loadError(path, store, "m").find("corrupted"), std::string::npos)
        << loadError(path, store, "m");

    IndexManager::save(idx, "m", path);
    fs::resize_file(path, fs::file_size(path) - 100);
    EXPECT_NE(loadError(path, store, "m").find("truncated"), std::string::npos)
        << loadError(path, store, "m");

    IndexManager::save(idx, "m", path);
    flipByte(path, 0);
    EXPECT_NE(loadError(path, store, "m").find("magic"), std::string::npos);

    IndexManager::save(idx, "m", path);
    EXPECT_NE(loadError(path, store, "other-model").find("stale"), std::string::npos);
    const auto otherStore = makeStore(600, 8, 4);  // same shape, different content
    EXPECT_NE(loadError(path, otherStore, "m").find("stale"), std::string::npos);
    const auto biggerStore = makeStore(601, 8, 3);
    EXPECT_NE(loadError(path, biggerStore, "m").find("stale"), std::string::npos);
    const auto otherDim = makeStore(600, 16, 3);
    EXPECT_NE(loadError(path, otherDim, "m").find("dimension mismatch"), std::string::npos);

    EXPECT_NE(loadError(tmpFile("does_not_exist.idx"), store, "m").find("not found"),
              std::string::npos);
}
