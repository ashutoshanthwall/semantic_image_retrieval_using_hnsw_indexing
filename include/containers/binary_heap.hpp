#pragma once
// Binary heap stored in an array: the children of slot i are 2i+1 and 2i+2, its parent is
// (i-1)/2. `less(a, b)` means "a ranks below b"; every parent ranks at least as high as
// its children, so the highest-ranked item is at slot 0 (a max-heap under `less`).
//
//   push  O(log n): append, then sift up while the parent ranks lower
//   pop   O(log n): move the last item to the root, then sift down towards the
//                   higher-ranked child
//   top   O(1)
//
// The sift routines are free functions so heap sort (algorithms/sort.hpp) shares them.
#include <cstddef>
#include <utility>

#include "containers/vector.hpp"

namespace sir {
namespace heap {

// Restores the heap order for a[i] after it was placed at slot i (a[0, i) is a heap).
template <typename T, typename Less>
void siftUp(T* a, std::size_t i, Less& less) {
    T item = std::move(a[i]);
    while (i > 0) {
        const std::size_t parent = (i - 1) / 2;
        if (!less(a[parent], item)) break;
        a[i] = std::move(a[parent]);  // parent moves down; the hole moves up
        i = parent;
    }
    a[i] = std::move(item);
}

// Restores the heap order of a[0, n) when only slot i may rank lower than a child.
template <typename T, typename Less>
void siftDown(T* a, std::size_t n, std::size_t i, Less& less) {
    T item = std::move(a[i]);
    for (;;) {
        std::size_t child = 2 * i + 1;
        if (child >= n) break;
        if (child + 1 < n && less(a[child], a[child + 1])) ++child;  // higher-ranked child
        if (!less(item, a[child])) break;
        a[i] = std::move(a[child]);  // child moves up; the hole moves down
        i = child;
    }
    a[i] = std::move(item);
}

}  // namespace heap

template <typename T, typename Less>
class BinaryHeap {
public:
    void reserve(std::size_t n) { items_.reserve(n); }
    std::size_t size() const noexcept { return items_.size(); }
    bool empty() const noexcept { return items_.empty(); }
    const T& top() const noexcept { return items_[0]; }

    void push(const T& item) {
        items_.push_back(item);
        heap::siftUp(items_.data(), items_.size() - 1, less_);
    }
    void pop() {
        if (items_.size() > 1) items_[0] = std::move(items_.back());
        items_.pop_back();
        if (!items_.empty()) heap::siftDown(items_.data(), items_.size(), 0, less_);
    }

private:
    Vector<T> items_;
    Less less_{};
};

}  // namespace sir
