#include <gtest/gtest.h>

#include <filesystem>
#include <limits>

#include "data/errors.hpp"
#include "embedding/text_encoder_process.hpp"
#include "index/brute_force_index.hpp"
#include "index/hnsw_index.hpp"
#include "search/search_engine.hpp"
#include "test_helpers.hpp"

using namespace sir;

namespace {

// Deterministic fake: returns a fixed (unnormalised) vector, or a bad one on request.
class FakeEncoder : public QueryEncoder {
public:
    explicit FakeEncoder(Vector<float> v) : v_(std::move(v)) {}
    Vector<float> encode(std::string_view text) override {
        lastText = std::string(text);
        return v_;
    }
    Vector<float> v_;
    std::string lastText;
};

struct World {
    static constexpr std::size_t N = 300, D = 16;
    EmbeddingStore store{D, test::randomUnitVectors(N, D, 21), 1e-5};
    MetadataManager meta{[] {
        Vector<ImageRecord> r;
        for (std::size_t i = 0; i < N; ++i)
            r.push_back({static_cast<int>(i), "img" + std::to_string(i),
                         "/data/img" + std::to_string(i) + ".jpg", 10, 10, "train", {"c"}});
        return r;
    }()};
    BruteForceIndex bf{store};
    HNSWIndex hnsw{store, [] {
        HnswParams p;
        p.M = 8;
        p.efConstruction = 64;
        p.efSearch = static_cast<int>(N);
        return p;
    }()};
    World() {
        bf.insertAll();
        for (std::size_t i = 0; i < N; ++i) hnsw.insert(static_cast<int>(i), store.row(i));
    }
};

}  // namespace

TEST(SearchEngine, NormalizesQueryText) {
    EXPECT_EQ(SearchEngine::normalizeQueryText("  a   dog\tin\n snow  "), "a dog in snow");
    EXPECT_THROW(SearchEngine::normalizeQueryText(""), QueryError);
    EXPECT_THROW(SearchEngine::normalizeQueryText(" \t\n "), QueryError);
}

TEST(SearchEngine, ResultsMapToMetadataAndAreBackendIndependent) {
    World w;
    auto raw = test::randomUnitVectors(1, World::D, 5);
    for (auto& x : raw) x *= 3.0f;  // unnormalised on purpose: engine must normalise
    FakeEncoder enc(raw);
    SearchEngine a(w.meta, w.store, w.bf, &enc), b(w.meta, w.store, w.hnsw, &enc);
    const auto ra = a.search("  a dog  ", 5), rb = b.search("a dog", 5);
    EXPECT_EQ(enc.lastText, "a dog");
    ASSERT_EQ(ra.results.size(), 5u);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(ra.results[i].rank, static_cast<int>(i) + 1);
        EXPECT_EQ(ra.results[i].internalId, rb.results[i].internalId);  // efSearch >= N
        EXPECT_EQ(ra.results[i].imageId, "img" + std::to_string(ra.results[i].internalId));
        EXPECT_GE(ra.results[i].score, -1.0f);
        EXPECT_LE(ra.results[i].score, 1.0f + 1e-6f);
    }
    EXPECT_GE(ra.retrievalMs, 0.0);
}

TEST(SearchEngine, ClearErrors) {
    World w;
    FakeEncoder enc(test::randomUnitVectors(1, World::D, 5));
    SearchEngine e(w.meta, w.store, w.bf, &enc);
    auto msg = [&](auto fn) {
        try {
            fn();
        } catch (const QueryError& ex) {
            return std::string(ex.what());
        }
        return std::string("<no error>");
    };
    EXPECT_NE(msg([&] { e.search("   ", 5); }).find("empty"), std::string::npos);
    EXPECT_NE(msg([&] { e.search("dog", 0); }).find("positive"), std::string::npos);
    EXPECT_NE(msg([&] { e.search("dog", -2); }).find("positive"), std::string::npos);
    EXPECT_NE(msg([&] { e.search("dog", 301); }).find("exceeds"), std::string::npos);
    enc.v_ = test::randomUnitVectors(1, 8, 1);
    EXPECT_NE(msg([&] { e.search("dog", 5); }).find("dimension mismatch"), std::string::npos);
    enc.v_ = test::randomUnitVectors(1, World::D, 1);
    enc.v_[3] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_NE(msg([&] { e.search("dog", 5); }).find("NaN"), std::string::npos);
    enc.v_[3] = std::numeric_limits<float>::infinity();
    EXPECT_NE(msg([&] { e.search("dog", 5); }).find("Inf"), std::string::npos);
    enc.v_.assign(World::D, 0.0f);
    EXPECT_NE(msg([&] { e.search("dog", 5); }).find("zero norm"), std::string::npos);
}

TEST(TextEncoderIntegration, RealHelperProcess) {
    const std::filesystem::path root(SIR_SOURCE_DIR);
    const auto py = root / ".venv/bin/python";
    if (!std::filesystem::exists(py)) GTEST_SKIP() << "no .venv";
    TextEncoderProcess enc(py, root / "scripts/text_encoder_server.py",
                           root / "config/config.json", 512);
    const auto v = enc.encode("a dog playing in snow");
    ASSERT_EQ(v.size(), 512u);
    EXPECT_TRUE(allFinite(v));
    EXPECT_GT(norm(v), 0.1);
    const auto v2 = enc.encode("a dog playing in snow");
    EXPECT_EQ(v, v2);  // deterministic across calls in one process
    EXPECT_THROW(enc.encode(""), QueryError);
}
