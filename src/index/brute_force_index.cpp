#include "index/brute_force_index.hpp"

#include <string>

#include "algorithms/basic.hpp"
#include "data/errors.hpp"
#include "similarity/vector_math.hpp"

namespace sir {

void checkInsert(const EmbeddingStore& store, int id, std::span<const float> embedding,
                 const BitVector& present) {
    if (id < 0 || static_cast<std::size_t>(id) >= store.size())
        throw IndexError("insert: id " + std::to_string(id) + " is not a row of the store (N=" +
                         std::to_string(store.size()) + ")");
    if (embedding.size() != store.dim())
        throw IndexError("insert: dimension mismatch for id " + std::to_string(id));
    const auto row = store.row(static_cast<std::size_t>(id));
    if (embedding.data() != row.data() && !equalRanges(row, embedding))
        throw IndexError("insert: embedding for id " + std::to_string(id) +
                         " differs from the store row (indexes reference the store)");
    if (present[static_cast<std::size_t>(id)])
        throw IndexError("insert: id " + std::to_string(id) + " already inserted");
}

BruteForceIndex::BruteForceIndex(const EmbeddingStore& store)
    : store_(&store), present_(store.size(), false) {}

void BruteForceIndex::insert(int id, std::span<const float> embedding) {
    checkInsert(*store_, id, embedding, present_);
    ids_.push_back(id);
    present_.set(static_cast<std::size_t>(id), true);
}

void BruteForceIndex::insertAll() {
    ids_.reserve(store_->size());
    for (std::size_t i = 0; i < store_->size(); ++i)
        insert(static_cast<int>(i), store_->row(i));
}

Vector<Candidate> BruteForceIndex::search(std::span<const float> query, int k) const {
    checkSearchArgs(query, k, store_->dim(), ids_.size());
    TopK top(static_cast<std::size_t>(k));
    for (int id : ids_) top.push({distance(query, store_->row(static_cast<std::size_t>(id))), id});
    return toCandidates(top.extractSorted());
}

std::size_t BruteForceIndex::memoryBytes() const {
    return ids_.capacity() * sizeof(int) + present_.capacity() / 8;
}

}  // namespace sir
