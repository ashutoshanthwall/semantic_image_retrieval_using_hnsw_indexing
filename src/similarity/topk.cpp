#include "similarity/topk.hpp"

namespace sir {

TopK::TopK(std::size_t k) : k_(k) { heap_.reserve(k + 1); }

void TopK::push(Neighbor n) {
    if (k_ == 0) return;
    if (heap_.size() < k_) {
        heap_.push(n);
    } else if (closer(n, heap_.top())) {  // beats the worst retained item
        heap_.pop();
        heap_.push(n);
    }
}

Vector<Neighbor> TopK::extractSorted() {
    Vector<Neighbor> out(heap_.size());
    for (std::size_t i = out.size(); i-- > 0;) {  // pops farthest first
        out[i] = heap_.top();
        heap_.pop();
    }
    return out;
}

}  // namespace sir
