#include "similarity/vector_math.hpp"

#include "data/errors.hpp"

namespace sir {

float dot(std::span<const float> a, std::span<const float> b) noexcept {
    float s = 0.0f;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

void normalize(std::span<float> a) {
    if (!allFinite(a)) throw EmbeddingError("cannot normalise a vector containing NaN/Inf");
    const double n = norm(a);
    if (n == 0.0) throw EmbeddingError("cannot normalise a zero-norm vector");
    for (float& x : a) x = static_cast<float>(static_cast<double>(x) / n);
}

}  // namespace sir
