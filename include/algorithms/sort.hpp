#pragma once
// Comparison sorting of a contiguous range, in place. Not stable.
//
// sortRange is an introsort:
//   1. quicksort with a median-of-three pivot and Hoare partitioning, recursing into the
//      smaller side and looping on the larger, so the stack depth is O(log n);
//   2. if the recursion goes deeper than 2*log2(n) (a run of bad pivots), that range is
//      finished with heap sort, so the worst case stays O(n log n);
//   3. ranges of at most 16 elements are left unsorted by the quicksort phase, and one
//      insertion-sort pass at the end fixes them; each element then moves at most 15
//      places, so that pass is O(n).
// `less` must be a strict weak ordering.
#include <cstddef>

#include "algorithms/basic.hpp"
#include "containers/binary_heap.hpp"

namespace sir {
namespace sort_detail {

constexpr std::size_t kInsertionThreshold = 16;

template <typename T, typename Less>
void insertionSort(T* a, std::size_t n, Less& less) {
    for (std::size_t i = 1; i < n; ++i) {
        T item = std::move(a[i]);
        std::size_t j = i;
        for (; j > 0 && less(item, a[j - 1]); --j) a[j] = std::move(a[j - 1]);
        a[j] = std::move(item);
    }
}

template <typename T, typename Less>
void heapSort(T* a, std::size_t n, Less& less) {
    for (std::size_t i = n / 2; i-- > 0;) heap::siftDown(a, n, i, less);  // build max-heap
    for (std::size_t end = n; end > 1; --end) {  // move the max behind the shrinking heap
        swapValues(a[0], a[end - 1]);
        heap::siftDown(a, end - 1, 0, less);
    }
}

// Hoare partition of a[0, n), n >= 3. Returns s with 0 < s < n such that every element of
// a[0, s) is <= every element of a[s, n).
template <typename T, typename Less>
std::size_t partition(T* a, std::size_t n, Less& less) {
    // Median of three: order a[0] <= a[mid] <= a[n-1]. The two outer samples then act as
    // sentinels that stop both scans before they can leave the range.
    const std::size_t mid = n / 2, last = n - 1;
    if (less(a[mid], a[0])) swapValues(a[mid], a[0]);
    if (less(a[last], a[mid])) {
        swapValues(a[last], a[mid]);
        if (less(a[mid], a[0])) swapValues(a[mid], a[0]);
    }
    const T pivot = a[mid];
    std::ptrdiff_t i = -1, j = static_cast<std::ptrdiff_t>(n);
    for (;;) {
        do ++i; while (less(a[i], pivot));
        do --j; while (less(pivot, a[j]));
        if (i >= j) return static_cast<std::size_t>(j) + 1;
        swapValues(a[i], a[j]);
    }
}

template <typename T, typename Less>
void introsortLoop(T* a, std::size_t n, int depthLeft, Less& less) {
    while (n > kInsertionThreshold) {
        if (depthLeft == 0) {
            heapSort(a, n, less);
            return;
        }
        --depthLeft;
        const std::size_t s = partition(a, n, less);
        if (s < n - s) {
            introsortLoop(a, s, depthLeft, less);
            a += s;
            n -= s;
        } else {
            introsortLoop(a + s, n - s, depthLeft, less);
            n = s;
        }
    }
}

}  // namespace sort_detail

template <typename T, typename Less>
void sortRange(T* first, T* last, Less less) {
    const auto n = static_cast<std::size_t>(last - first);
    if (n < 2) return;
    int log2n = 0;
    for (std::size_t m = n; m > 1; m /= 2) ++log2n;
    sort_detail::introsortLoop(first, n, 2 * log2n, less);
    sort_detail::insertionSort(first, n, less);
}

// Ascending order by operator<.
template <typename T>
void sortRange(T* first, T* last) {
    sortRange(first, last, [](const T& a, const T& b) { return a < b; });
}

}  // namespace sir
