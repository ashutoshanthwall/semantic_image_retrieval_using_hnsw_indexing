#pragma once
// Small generic helpers used across the engine in place of <algorithm>.
#include <cstddef>
#include <span>
#include <utility>

namespace sir {

template <typename T>
constexpr T minOf(T a, T b) {
    return b < a ? b : a;
}

template <typename T>
constexpr T maxOf(T a, T b) {
    return a < b ? b : a;
}

template <typename T>
void swapValues(T& a, T& b) {
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

// True if both ranges have the same length and equal elements in the same order.
template <typename T>
bool equalRanges(std::span<const T> a, std::span<const T> b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!(a[i] == b[i])) return false;
    return true;
}

}  // namespace sir
