#include <gtest/gtest.h>

#include <random>

#include "algorithms/basic.hpp"
#include "algorithms/sort.hpp"
#include "similarity/topk.hpp"
#include "similarity/vector_math.hpp"
#include "test_helpers.hpp"

using namespace sir;

namespace {
Vector<Neighbor> fullSort(Vector<Neighbor> all, std::size_t k) {
    sortRange(all.begin(), all.end(), closer);
    all.resize(minOf(k, all.size()));
    return all;
}
}  // namespace

TEST(TopK, SmallHandExample) {
    TopK t(3);
    for (auto n : {Neighbor{0.5f, 4}, Neighbor{0.1f, 9}, Neighbor{0.9f, 1}, Neighbor{0.1f, 2},
                   Neighbor{0.3f, 7}})
        t.push(n);
    const auto r = t.extractSorted();
    ASSERT_EQ(r.size(), 3u);
    EXPECT_EQ(r[0].id, 2);  // tie at 0.1 broken by ascending id
    EXPECT_EQ(r[1].id, 9);
    EXPECT_EQ(r[2].id, 7);
}

TEST(TopK, KLargerThanN) {
    TopK t(10);
    t.push({0.2f, 1});
    t.push({0.1f, 0});
    const auto r = t.extractSorted();
    ASSERT_EQ(r.size(), 2u);
    EXPECT_EQ(r[0].id, 0);
}

// G6: top-K against a full sort on 1000 random vectors, 100 trials, exact match
// including tie order. Ties are injected deliberately by duplicating vectors.
TEST(TopK, MatchesFullSortWithTies_G6) {
    constexpr std::size_t N = 1000, D = 32;
    std::mt19937_64 rng(123);
    int trials = 0, exact = 0;
    for (int trial = 0; trial < 100; ++trial) {
        auto data = test::randomUnitVectors(N, D, 1000 + static_cast<std::uint64_t>(trial));
        // Make ~5% of rows exact duplicates of other rows -> exact distance ties.
        std::uniform_int_distribution<std::size_t> pick(0, N - 1);
        for (int d = 0; d < 50; ++d) {
            const std::size_t src = pick(rng), dst = pick(rng);
            for (std::size_t j = 0; j < D; ++j) data[dst * D + j] = data[src * D + j];
        }
        const auto q = test::randomUnitVectors(1, D, 5000 + static_cast<std::uint64_t>(trial));
        const std::size_t k = 1 + static_cast<std::size_t>(trial % 50);
        Vector<Neighbor> all;
        TopK t(k);
        for (std::size_t i = 0; i < N; ++i) {
            const Neighbor n{distance(q, std::span<const float>(data.data() + i * D, D)),
                             static_cast<int>(i)};
            all.push_back(n);
            t.push(n);
        }
        const auto got = t.extractSorted();
        const auto want = fullSort(all, k);
        ++trials;
        ASSERT_EQ(got.size(), want.size());
        bool same = true;
        for (std::size_t i = 0; i < k; ++i)
            same &= got[i].id == want[i].id && got[i].dist == want[i].dist;
        exact += same;
        ASSERT_TRUE(same) << "trial " << trial;
    }
    std::cout << "[G6] top-K vs full sort: " << exact << "/" << trials << " trials exact\n";
}
