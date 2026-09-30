#include "similarity/vector_math.hpp"

#include "data/errors.hpp"

namespace sir {

void normalize(std::span<float> a) {
    if (!allFinite(a)) throw EmbeddingError("cannot normalise a vector containing NaN/Inf");
    const double n = norm(a);
    if (n == 0.0) throw EmbeddingError("cannot normalise a zero-norm vector");
    for (float& x : a) x = static_cast<float>(static_cast<double>(x) / n);
}

}  // namespace sir
