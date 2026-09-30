#pragma once
#include <cstdint>
#include <random>
#include <vector>

#include "similarity/vector_math.hpp"

namespace sir::test {

// n random unit vectors of dimension d (Gaussian, then normalised), row-major.
inline std::vector<float> randomUnitVectors(std::size_t n, std::size_t d, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> v(n * d);
    for (auto& x : v) x = g(rng);
    for (std::size_t i = 0; i < n; ++i) normalize(std::span<float>(v.data() + i * d, d));
    return v;
}

}  // namespace sir::test
