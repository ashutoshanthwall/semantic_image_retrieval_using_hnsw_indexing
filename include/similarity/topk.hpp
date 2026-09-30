#pragma once
// Bounded top-K selection. Keeps the K closest (lowest distance) items seen so far in a
// max-heap whose top is the *worst* retained item, so each push is O(log K) and the
// whole scan is O(N log K) instead of the O(N log N) of a full sort.
//
// Ordering is total and deterministic: by distance, then by ascending id. Flickr30k has
// near-duplicate images, so exact distance ties happen; the id tie-break makes results
// identical across backends and runs.
#include <cstddef>
#include <queue>
#include <vector>

namespace sir {

struct Neighbor {
    float dist;
    int id;
};

// Strict weak ordering "a is better (closer) than b".
inline bool closer(const Neighbor& a, const Neighbor& b) noexcept {
    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
}

// std::priority_queue puts the "largest" element on top. With `closer` as the
// less-than, the top is the farthest element: a result max-heap.
struct FarthestOnTop {
    bool operator()(const Neighbor& a, const Neighbor& b) const noexcept { return closer(a, b); }
};
// Reversed: the top is the closest element: a candidate min-heap.
struct ClosestOnTop {
    bool operator()(const Neighbor& a, const Neighbor& b) const noexcept { return closer(b, a); }
};

using MaxDistHeap = std::priority_queue<Neighbor, std::vector<Neighbor>, FarthestOnTop>;
using MinDistHeap = std::priority_queue<Neighbor, std::vector<Neighbor>, ClosestOnTop>;

class TopK {
public:
    explicit TopK(std::size_t k);

    // Offers a candidate; retained only if fewer than k are held or it beats the worst.
    void push(Neighbor n);
    std::size_t size() const noexcept { return heap_.size(); }
    std::size_t capacity() const noexcept { return k_; }
    // Closest first. Consumes the heap.
    std::vector<Neighbor> extractSorted();

private:
    std::size_t k_;
    MaxDistHeap heap_;
};

}  // namespace sir
