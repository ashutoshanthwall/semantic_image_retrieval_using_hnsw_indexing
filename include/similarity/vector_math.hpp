#pragma once
// Vector math on float32 embeddings. Deliberately plain scalar loops: no intrinsics, no
// BLAS, no OpenMP. Both retrieval backends call exactly these functions, so the
// brute-force vs HNSW comparison measures the algorithms, not optimisation effort.
//
// Distance convention used everywhere inside the engine (queues, comparators, pruning):
//     dist(a, b) = 1 - dot(a, b)   on unit vectors; lower is better.
// Similarity (higher is better) exists only at the output boundary (Candidate::score).
#include <cmath>
#include <cstddef>
#include <span>

namespace sir {

// Precondition: a.size() == b.size() (checked by callers at API boundaries).
// Defined out of line on purpose: an inline version is compiled separately at every call
// site, and the compiler may contract (FMA) or schedule it differently per site, giving
// 1-ulp differences between backends. One compiled instance makes every distance in the
// system bit-identical for the same pair of vectors, which exact tie-breaking and the
// "HNSW == brute force at high ef" gate rely on.
float dot(std::span<const float> a, std::span<const float> b) noexcept;

inline float distance(std::span<const float> a, std::span<const float> b) noexcept {
    return 1.0f - dot(a, b);
}

inline float distanceToScore(float dist) noexcept { return 1.0f - dist; }

// Accumulated in double: used for validation / normalisation, not in search loops.
inline double squaredNorm(std::span<const float> a) noexcept {
    double s = 0.0;
    for (float x : a) s += static_cast<double>(x) * static_cast<double>(x);
    return s;
}

inline double norm(std::span<const float> a) noexcept { return std::sqrt(squaredNorm(a)); }

inline bool allFinite(std::span<const float> a) noexcept {
    for (float x : a)
        if (!std::isfinite(x)) return false;
    return true;
}

// Scales `a` to unit L2 norm in place. Throws EmbeddingError on zero-norm or
// non-finite input (a vector that cannot be normalised must never reach an index).
void normalize(std::span<float> a);

}  // namespace sir
