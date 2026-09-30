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
inline float dot(std::span<const float> a, std::span<const float> b) noexcept {
    float s = 0.0f;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

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
