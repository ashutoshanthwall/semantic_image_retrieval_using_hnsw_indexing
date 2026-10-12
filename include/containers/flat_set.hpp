#pragma once
// FlatSet<T>: a set kept as a sorted Vector without duplicates. Lookup is a binary search,
// O(log n); insertion shifts the larger elements right, O(n). The sets here hold a few
// dozen ids at most (top-k results, one node's neighbour list), where a contiguous sorted
// array beats a node-based tree on both speed and memory. Iterates in ascending order.
// T needs operator<.
#include <cstddef>

#include "containers/vector.hpp"

namespace sir {

template <typename T>
class FlatSet {
public:
    // Returns true if `value` was added, false if it was already present.
    bool insert(const T& value) {
        const std::size_t pos = lowerBound(value);
        if (pos < items_.size() && !(value < items_[pos])) return false;
        items_.insertAt(pos, value);
        return true;
    }
    bool contains(const T& value) const {
        const std::size_t pos = lowerBound(value);
        return pos < items_.size() && !(value < items_[pos]);
    }
    std::size_t count(const T& value) const { return contains(value) ? 1 : 0; }

    std::size_t size() const noexcept { return items_.size(); }
    bool empty() const noexcept { return items_.empty(); }
    const T* begin() const noexcept { return items_.begin(); }
    const T* end() const noexcept { return items_.end(); }

private:
    // First position whose element is not less than `value` (size() if none).
    std::size_t lowerBound(const T& value) const {
        std::size_t lo = 0, hi = items_.size();
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (items_[mid] < value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    Vector<T> items_;
};

}  // namespace sir
