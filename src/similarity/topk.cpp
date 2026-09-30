#include "similarity/topk.hpp"

#include <algorithm>

namespace sir {

TopK::TopK(std::size_t k) : k_(k) {
    std::vector<Neighbor> storage;
    storage.reserve(k + 1);
    heap_ = MaxDistHeap(FarthestOnTop{}, std::move(storage));
}

void TopK::push(Neighbor n) {
    if (k_ == 0) return;
    if (heap_.size() < k_) {
        heap_.push(n);
    } else if (closer(n, heap_.top())) {  // beats the worst retained item
        heap_.pop();
        heap_.push(n);
    }
}

std::vector<Neighbor> TopK::extractSorted() {
    std::vector<Neighbor> out;
    out.reserve(heap_.size());
    while (!heap_.empty()) {  // pops farthest first
        out.push_back(heap_.top());
        heap_.pop();
    }
    std::reverse(out.begin(), out.end());
    return out;
}

}  // namespace sir
