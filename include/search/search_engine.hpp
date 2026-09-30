#pragma once
// SearchEngine orchestrates: query text validation/normalisation -> text embedding ->
// vector validation/normalisation -> RetrievalIndex::search -> metadata lookup. It knows
// nothing about which index it is talking to; SearchResult is algorithm-independent.
//
// Ownership: MetadataManager owns records, EmbeddingStore owns vectors, the index owns its
// own structure; SearchEngine holds non-owning references to all of them, and they must
// outlive it.
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "data/metadata_manager.hpp"
#include "embedding/embedding_store.hpp"
#include "embedding/query_encoder.hpp"
#include "index/retrieval_index.hpp"

namespace sir {

struct SearchResult {
    int rank;
    int internalId;
    std::string imageId;
    std::filesystem::path path;
    float score;
};

struct SearchResponse {
    std::vector<SearchResult> results;
    double retrievalMs = 0.0;  // RetrievalIndex::search only
};

class SearchEngine {
public:
    SearchEngine(const MetadataManager& metadata, const EmbeddingStore& store,
                 const RetrievalIndex& index, QueryEncoder* encoder);

    // Trims and collapses whitespace; throws QueryError for empty/whitespace-only text.
    static std::string normalizeQueryText(std::string_view text);
    // Validates K against the index before any expensive work.
    void checkK(int k) const;
    // Text -> unit-norm query vector (validated: dimension, NaN/Inf, zero norm).
    std::vector<float> embedQuery(std::string_view text) const;
    std::vector<float> prepareVector(std::span<const float> raw) const;

    SearchResponse search(std::string_view text, int k) const;
    SearchResponse searchEmbedding(std::span<const float> unitQuery, int k) const;

private:
    const MetadataManager* metadata_;
    const EmbeddingStore* store_;
    const RetrievalIndex* index_;
    QueryEncoder* encoder_;  // may be null for precomputed-embedding use
};

}  // namespace sir
