#pragma once
// Backend-agnostic retrieval interface implemented by BruteForceIndex and HNSWIndex.
#include <cstddef>
#include <span>
#include <vector>

#include "similarity/topk.hpp"

namespace sir {

struct Candidate {
    int id;
    float score;  // similarity, higher is better
};

class RetrievalIndex {
public:
    virtual void insert(int id, std::span<const float> embedding) = 0;
    // Top-k by descending score; ties broken by ascending id (deterministic).
    virtual std::vector<Candidate> search(std::span<const float> query, int k) const = 0;
    virtual std::size_t size() const = 0;
    virtual std::size_t memoryBytes() const = 0;
    virtual ~RetrievalIndex() = default;
};

// Output boundary: converts closest-first (dist, id) neighbours into similarity-scored
// candidates. score = 1 - dist; because that float conversion can map two distinct
// distances to one score, the final order is re-established as (score desc, id asc).
std::vector<Candidate> toCandidates(const std::vector<Neighbor>& closestFirst);

// Argument checks shared by both backends: dimension, finiteness, 0 < k <= size.
void checkSearchArgs(std::span<const float> query, int k, std::size_t dim, std::size_t size);

}  // namespace sir
