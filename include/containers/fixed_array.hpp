#pragma once
// FixedArray<T, N>: a C array of N elements wrapped in a struct, so it can be returned by
// value, copied, and passed as a std::span. It is an aggregate: FixedArray<int, 3> a{1, 2, 3}.
#include <cstddef>

namespace sir {

template <typename T, std::size_t N>
struct FixedArray {
    T items[N];

    constexpr std::size_t size() const noexcept { return N; }
    constexpr T& operator[](std::size_t i) noexcept { return items[i]; }
    constexpr const T& operator[](std::size_t i) const noexcept { return items[i]; }
    constexpr T* data() noexcept { return items; }
    constexpr const T* data() const noexcept { return items; }
    constexpr T* begin() noexcept { return items; }
    constexpr T* end() noexcept { return items + N; }
    constexpr const T* begin() const noexcept { return items; }
    constexpr const T* end() const noexcept { return items + N; }
};

}  // namespace sir
