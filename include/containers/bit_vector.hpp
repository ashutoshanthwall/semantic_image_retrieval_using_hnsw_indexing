#pragma once
// BitVector: a fixed-length array of booleans packed 64 per word (1 bit per flag instead
// of 1 byte). Used for "already inserted" / "already visited" markers over node ids.
#include <cstddef>
#include <cstdint>

#include "containers/vector.hpp"

namespace sir {

class BitVector {
public:
    BitVector() = default;
    BitVector(std::size_t n, bool value)
        : words_((n + 63) / 64, value ? ~std::uint64_t{0} : std::uint64_t{0}), size_(n) {}

    std::size_t size() const noexcept { return size_; }
    // Allocated bits (whole words), for memory accounting.
    std::size_t capacity() const noexcept { return words_.capacity() * 64; }

    bool operator[](std::size_t i) const noexcept { return (words_[i / 64] >> (i % 64)) & 1u; }
    void set(std::size_t i, bool value) noexcept {
        const std::uint64_t bit = std::uint64_t{1} << (i % 64);
        if (value) words_[i / 64] |= bit;
        else words_[i / 64] &= ~bit;
    }

private:
    Vector<std::uint64_t> words_;
    std::size_t size_ = 0;
};

}  // namespace sir
