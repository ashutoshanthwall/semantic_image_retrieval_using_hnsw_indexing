#pragma once
// Exact baseline and ground truth: scores every inserted vector against the query and
// keeps the best k in a bounded heap (O(N log k)). Single-threaded, scalar.
#include "containers/bit_vector.hpp"
#include "containers/vector.hpp"
#include "embedding/embedding_store.hpp"
#include "index/retrieval_index.hpp"

namespace sir {

class BruteForceIndex final : public RetrievalIndex {
public:
    explicit BruteForceIndex(const EmbeddingStore& store);

    // `embedding` must be the store's row for `id` (the index keeps only the id).
    void insert(int id, std::span<const float> embedding) override;
    void insertAll();  // every store row, in id order
    Vector<Candidate> search(std::span<const float> query, int k) const override;
    std::size_t size() const override { return ids_.size(); }
    std::size_t memoryBytes() const override;

private:
    const EmbeddingStore* store_;  // non-owning; the store outlives the index
    Vector<int> ids_;
    BitVector present_;
};

// Checks that an insert refers to a valid store row and that the caller's span holds
// exactly that row's values; throws IndexError otherwise. Shared by both backends.
void checkInsert(const EmbeddingStore& store, int id, std::span<const float> embedding,
                 const BitVector& present);

}  // namespace sir
