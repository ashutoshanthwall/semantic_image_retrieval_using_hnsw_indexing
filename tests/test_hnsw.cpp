#include <gtest/gtest.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <queue>
#include <set>

#include "data/errors.hpp"
#include "index/brute_force_index.hpp"
#include "index/hnsw_index.hpp"
#include "test_helpers.hpp"

using namespace sir;

namespace {

struct Fixture {
    EmbeddingStore store;
    BruteForceIndex bf;
    HNSWIndex hnsw;
    Fixture(std::size_t n, std::size_t d, HnswParams p, std::uint64_t seed)
        : store(d, test::randomUnitVectors(n, d, seed), 1e-5), bf(store), hnsw(store, p) {
        bf.insertAll();
        for (std::size_t i = 0; i < n; ++i) hnsw.insert(static_cast<int>(i), store.row(i));
    }
};

HnswParams params(int M, int efC, int efS, bool heuristic) {
    HnswParams p;
    p.M = M;
    p.efConstruction = efC;
    p.efSearch = efS;
    p.seed = 42;
    p.selectHeuristic = heuristic;
    return p;
}

// Checks graph invariants: degree caps, no self loops, no duplicate links, links point to
// present nodes that exist on that layer, every node reachable from the entry on layer 0.
void checkInvariants(const HNSWIndex& h, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const int id = static_cast<int>(i);
        ASSERT_TRUE(h.contains(id));
        for (int l = 0; l <= h.levelOf(id); ++l) {
            const auto nb = h.neighbors(id, l);
            ASSERT_LE(nb.size(), static_cast<std::size_t>(h.maxDegree(l)));
            std::set<int> uniq(nb.begin(), nb.end());
            ASSERT_EQ(uniq.size(), nb.size()) << "duplicate link at node " << id;
            for (int e : nb) {
                ASSERT_NE(e, id) << "self loop";
                ASSERT_GE(h.levelOf(e), l) << "link to a node absent from layer " << l;
            }
        }
    }
    // Incremental in-degree bookkeeping must equal a full recount.
    for (int l = 0; l <= h.maxLevel(); ++l) {
        std::vector<std::uint32_t> cnt(n, 0);
        for (std::size_t i = 0; i < n; ++i)
            if (h.levelOf(static_cast<int>(i)) >= l)
                for (int e : h.neighbors(static_cast<int>(i), l)) ++cnt[static_cast<std::size_t>(e)];
        for (std::size_t i = 0; i < n; ++i)
            if (h.levelOf(static_cast<int>(i)) >= l)
                ASSERT_EQ(cnt[i], h.inDegreeOf(static_cast<int>(i), l)) << "node " << i << " layer " << l;
    }
    std::vector<bool> seen(n, false);
    std::queue<int> bfs;
    bfs.push(h.entryPoint());
    seen[static_cast<std::size_t>(h.entryPoint())] = true;
    std::size_t reached = 1;
    while (!bfs.empty()) {
        const int c = bfs.front();
        bfs.pop();
        for (int e : h.neighbors(c, 0))
            if (!seen[static_cast<std::size_t>(e)]) {
                seen[static_cast<std::size_t>(e)] = true;
                ++reached;
                bfs.push(e);
            }
    }
    EXPECT_EQ(reached, n) << "layer-0 graph is disconnected";
}

void expectExactWithHighEf(std::size_t n, std::size_t d, int k, bool heuristic) {
    Fixture f(n, d, params(4, 16, static_cast<int>(n), heuristic), 99 + n);
    checkInvariants(f.hnsw, n);
    const auto qs = test::randomUnitVectors(50, d, 7 + n);
    int exact = 0;
    for (int t = 0; t < 50; ++t) {
        std::span<const float> q(qs.data() + static_cast<std::size_t>(t) * d, d);
        const auto a = f.bf.search(q, k);
        const auto b = f.hnsw.search(q, k);
        ASSERT_EQ(a.size(), b.size());
        bool same = true;
        for (std::size_t i = 0; i < a.size(); ++i)
            same &= a[i].id == b[i].id && a[i].score == b[i].score;
        exact += same;
        EXPECT_TRUE(same) << "query " << t;
    }
    // Also every stored vector as its own query.
    for (std::size_t i = 0; i < n; ++i) {
        const auto a = f.bf.search(f.store.row(i), k);
        const auto b = f.hnsw.search(f.store.row(i), k);
        for (std::size_t r = 0; r < a.size(); ++r) ASSERT_EQ(a[r].id, b[r].id);
    }
    std::cout << "[G8] N=" << n << " D=" << d << " k=" << k << " efSearch=" << n
              << (heuristic ? " heuristic" : " naive") << ": " << exact
              << "/50 random queries identical to brute force (+" << n
              << " self-queries)\n";
}

}  // namespace

TEST(HNSW, G8_Exact_N10_D8) {
    expectExactWithHighEf(10, 8, 5, true);
    expectExactWithHighEf(10, 8, 10, true);
    expectExactWithHighEf(10, 8, 10, false);
}

TEST(HNSW, G8_Exact_N100_D8) {
    expectExactWithHighEf(100, 8, 10, true);
    expectExactWithHighEf(100, 8, 10, false);
}

TEST(HNSW, G8_Recall_N5000) {
    constexpr std::size_t N = 5000, D = 32;
    for (bool heuristic : {true, false}) {
        Fixture f(N, D, params(16, 200, 20, heuristic), 2024);
        checkInvariants(f.hnsw, N);
        const auto qs = test::randomUnitVectors(200, D, 77);
        std::cout << "[G8] N=5000 D=32 M=16 efC=200 "
                  << (heuristic ? "heuristic" : "naive    ") << " Recall@10:";
        double prev = 0.0;
        for (int ef : {20, 40, 80, 120}) {
            f.hnsw.setEfSearch(ef);
            double hits = 0;
            for (int t = 0; t < 200; ++t) {
                std::span<const float> q(qs.data() + static_cast<std::size_t>(t) * D, D);
                const auto a = f.bf.search(q, 10), b = f.hnsw.search(q, 10);
                std::set<int> truth;
                for (auto& c : a) truth.insert(c.id);
                for (auto& c : b) hits += truth.count(c.id);
            }
            const double recall = hits / (200.0 * 10.0);
            std::cout << "  ef=" << ef << ":" << std::fixed << std::setprecision(4) << recall;
            EXPECT_GE(recall + 1e-12, prev) << "recall decreased as efSearch grew";
            prev = recall;
        }
        std::cout << "\n";
    }
}

TEST(HNSW, DeterministicForFixedSeed) {
    Fixture a(800, 16, params(8, 64, 32, true), 5), b(800, 16, params(8, 64, 32, true), 5);
    ASSERT_EQ(a.hnsw.entryPoint(), b.hnsw.entryPoint());
    for (int i = 0; i < 800; ++i) {
        ASSERT_EQ(a.hnsw.levelOf(i), b.hnsw.levelOf(i));
        for (int l = 0; l <= a.hnsw.levelOf(i); ++l) {
            const auto x = a.hnsw.neighbors(i, l), y = b.hnsw.neighbors(i, l);
            ASSERT_TRUE(std::equal(x.begin(), x.end(), y.begin(), y.end()));
        }
    }
}

TEST(HNSW, LevelDistributionFollowsExponentialDecay) {
    Fixture f(5000, 8, params(16, 32, 16, true), 3);
    std::vector<int> hist(8, 0);
    for (int i = 0; i < 5000; ++i) ++hist[static_cast<std::size_t>(std::min(f.hnsw.levelOf(i), 7))];
    // P(level >= 1) = 1/M = 0.0625 -> expect ~312 of 5000; allow a wide binomial band.
    const int upper = 5000 - hist[0];
    EXPECT_GT(upper, 240);
    EXPECT_LT(upper, 390);
}

TEST(HNSW, ErrorsAndParams) {
    EmbeddingStore store(4, test::randomUnitVectors(20, 4, 1), 1e-5);
    EXPECT_THROW(HNSWIndex(store, params(1, 10, 10, true)), IndexError);
    HNSWIndex h(store, params(4, 10, 10, true));
    for (int i = 0; i < 20; ++i) h.insert(i, store.row(static_cast<std::size_t>(i)));
    const auto q = test::randomUnitVectors(1, 4, 9);
    EXPECT_THROW(h.search(q, 0), QueryError);
    EXPECT_THROW(h.search(q, 21), QueryError);
    EXPECT_THROW(h.insert(3, store.row(3)), IndexError);
    EXPECT_EQ(h.search(q, 20).size(), 20u);  // k larger than efSearch -> ef = k
}
