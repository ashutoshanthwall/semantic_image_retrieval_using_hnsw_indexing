#include <gtest/gtest.h>

#include <algorithm>
#include <limits>

#include "data/errors.hpp"
#include "index/brute_force_index.hpp"
#include "test_helpers.hpp"

using namespace sir;

TEST(BruteForce, MatchesFullSortOrderAndScores) {
    const std::size_t N = 500, D = 16;
    EmbeddingStore store(D, test::randomUnitVectors(N, D, 11), 1e-5);
    BruteForceIndex bf(store);
    bf.insertAll();
    const auto q = test::randomUnitVectors(20, D, 12);
    for (int t = 0; t < 20; ++t) {
        std::span<const float> qs(q.data() + t * D, D);
        const auto got = bf.search(qs, 10);
        std::vector<Candidate> all;
        for (std::size_t i = 0; i < N; ++i)
            all.push_back({static_cast<int>(i), 1.0f - (1.0f - dot(qs, store.row(i)))});
        std::sort(all.begin(), all.end(), [](auto& a, auto& b) {
            return a.score > b.score || (a.score == b.score && a.id < b.id);
        });
        for (int r = 0; r < 10; ++r) {
            EXPECT_EQ(got[r].id, all[r].id);
            EXPECT_EQ(got[r].score, all[r].score);
        }
    }
}

TEST(BruteForce, ExactTiesBreakByAscendingId) {
    std::vector<float> v = test::randomUnitVectors(6, 4, 3);
    for (int dup : {4, 1, 5})  // rows 1, 4, 5 identical to row 2
        std::copy_n(v.begin() + 2 * 4, 4, v.begin() + dup * 4);
    EmbeddingStore store(4, std::move(v), 1e-5);
    BruteForceIndex bf(store);
    bf.insertAll();
    const auto r = bf.search(store.row(2), 4);
    EXPECT_EQ(r[0].id, 1);
    EXPECT_EQ(r[1].id, 2);
    EXPECT_EQ(r[2].id, 4);
    EXPECT_EQ(r[3].id, 5);
    EXPECT_EQ(r[0].score, r[3].score);
}

TEST(BruteForce, ArgumentErrors) {
    EmbeddingStore store(4, test::randomUnitVectors(5, 4, 1), 1e-5);
    BruteForceIndex bf(store);
    bf.insertAll();
    const auto q = test::randomUnitVectors(1, 4, 2);
    EXPECT_THROW(bf.search(q, 0), QueryError);
    EXPECT_THROW(bf.search(q, -3), QueryError);
    EXPECT_THROW(bf.search(q, 6), QueryError);
    EXPECT_NO_THROW(bf.search(q, 5));
    const auto q8 = test::randomUnitVectors(1, 8, 2);
    EXPECT_THROW(bf.search(q8, 1), QueryError);
    auto bad = q;
    bad[1] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(bf.search(bad, 1), QueryError);
    bad[1] = std::numeric_limits<float>::infinity();
    EXPECT_THROW(bf.search(bad, 1), QueryError);
    EXPECT_THROW(bf.insert(0, store.row(0)), IndexError);  // duplicate insert
    EXPECT_THROW(bf.insert(9, store.row(0)), IndexError);  // not a store row
    EXPECT_THROW(bf.insert(1, store.row(2)), IndexError);  // wrong vector for id
}
