#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "data/errors.hpp"
#include "similarity/vector_math.hpp"
#include "test_helpers.hpp"

using namespace sir;

TEST(VectorMath, DotHandComputed) {
    const std::vector<float> a{1, 2, 3, 4}, b{5, 6, 7, 8};
    EXPECT_FLOAT_EQ(dot(a, b), 70.0f);  // 5+12+21+32
    const std::vector<float> c{0.5f, -0.25f}, d{-2.0f, 4.0f};
    EXPECT_FLOAT_EQ(dot(c, d), -2.0f);  // -1 - 1
    EXPECT_FLOAT_EQ(dot(a, a), 30.0f);
}

TEST(VectorMath, NormHandComputed) {
    const std::vector<float> a{3, 4};
    EXPECT_DOUBLE_EQ(norm(a), 5.0);
    EXPECT_DOUBLE_EQ(squaredNorm(a), 25.0);
    const std::vector<float> b{1, 2, 2};
    EXPECT_DOUBLE_EQ(norm(b), 3.0);
}

TEST(VectorMath, DistanceConvention) {
    const std::vector<float> e1{1, 0}, e2{0, 1}, m1{-1, 0};
    EXPECT_FLOAT_EQ(distance(e1, e1), 0.0f);  // identical -> 0
    EXPECT_FLOAT_EQ(distance(e1, e2), 1.0f);  // orthogonal -> 1
    EXPECT_FLOAT_EQ(distance(e1, m1), 2.0f);  // opposite -> 2
    EXPECT_FLOAT_EQ(distanceToScore(distance(e1, e2)), 0.0f);
}

TEST(VectorMath, NormalizeHandComputed) {
    std::vector<float> a{3, 4};
    normalize(a);
    EXPECT_FLOAT_EQ(a[0], 0.6f);
    EXPECT_FLOAT_EQ(a[1], 0.8f);
}

TEST(VectorMath, NormalizeIsIdempotent) {
    auto v = test::randomUnitVectors(200, 512, 7);
    for (std::size_t i = 0; i < 200; ++i) {
        std::span<float> row(v.data() + i * 512, 512);
        std::vector<float> once(row.begin(), row.end());
        std::vector<float> twice = once;
        normalize(twice);
        EXPECT_NEAR(norm(once), 1.0, 1e-6);
        for (std::size_t j = 0; j < 512; ++j)
            ASSERT_NEAR(once[j], twice[j], 1e-7) << "row " << i << " elem " << j;
    }
}

TEST(VectorMath, NormalizeRejectsBadInput) {
    std::vector<float> z(8, 0.0f);
    EXPECT_THROW(normalize(z), EmbeddingError);
    std::vector<float> n{1.0f, std::numeric_limits<float>::quiet_NaN()};
    EXPECT_THROW(normalize(n), EmbeddingError);
    std::vector<float> inf{1.0f, std::numeric_limits<float>::infinity()};
    EXPECT_THROW(normalize(inf), EmbeddingError);
}
