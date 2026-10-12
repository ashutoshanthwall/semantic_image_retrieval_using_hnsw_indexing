#include <gtest/gtest.h>

#include <limits>

#include "data/errors.hpp"
#include "embedding/embedding_store.hpp"
#include "test_helpers.hpp"

using namespace sir;

TEST(EmbeddingStore, RowsAreViewsIntoOwnedBuffer) {
    auto v = test::randomUnitVectors(10, 8, 1);
    const Vector<float> copy = v;
    EmbeddingStore s(8, std::move(v), 1e-5);
    EXPECT_EQ(s.size(), 10u);
    EXPECT_EQ(s.dim(), 8u);
    EXPECT_EQ(s.row(3)[5], copy[3 * 8 + 5]);
    EXPECT_EQ(s.row(1).data() + 8, s.row(2).data());  // contiguous, no per-row copies
}

TEST(EmbeddingStore, RejectsInvalidRowsNamingTheRow) {
    auto make = [](float bad, std::size_t row) {
        auto v = test::randomUnitVectors(5, 4, 2);
        v[row * 4 + 1] = bad;
        return v;
    };
    try {
        EmbeddingStore s(4, make(std::numeric_limits<float>::quiet_NaN(), 3), 1e-5);
        FAIL() << "expected throw";
    } catch (const EmbeddingError& e) {
        EXPECT_NE(std::string(e.what()).find("row 3"), std::string::npos) << e.what();
        EXPECT_NE(std::string(e.what()).find("NaN"), std::string::npos);
    }
    EXPECT_THROW(EmbeddingStore(4, make(std::numeric_limits<float>::infinity(), 1), 1e-5),
                 EmbeddingError);
    Vector<float> zero(8, 0.0f);
    EXPECT_THROW(EmbeddingStore(4, zero, 1e-5), EmbeddingError);
    auto unnorm = test::randomUnitVectors(2, 4, 3);
    unnorm[0] *= 1.01f;
    EXPECT_THROW(EmbeddingStore(4, unnorm, 1e-5), EmbeddingError);
    EXPECT_THROW(EmbeddingStore(4, Vector<float>(7, 0.5f), 1e-5), EmbeddingError);
}

TEST(EmbeddingStore, ValidateVectorDimensionMismatch) {
    const auto v = test::randomUnitVectors(1, 8, 4);
    EXPECT_THROW(validateVector(v, 16, 1e-5, "query"), EmbeddingError);
    EXPECT_NO_THROW(validateVector(v, 8, 1e-5, "query"));
}
