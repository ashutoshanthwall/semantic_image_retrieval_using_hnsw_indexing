#include "search/search_engine.hpp"

#include <cctype>
#include <chrono>
#include <cmath>

#include "data/errors.hpp"
#include "similarity/vector_math.hpp"

namespace sir {

SearchEngine::SearchEngine(const MetadataManager& metadata, const EmbeddingStore& store,
                           const RetrievalIndex& index, QueryEncoder* encoder)
    : metadata_(&metadata), store_(&store), index_(&index), encoder_(encoder) {
    if (metadata.size() != store.size())
        throw DataError("metadata has " + std::to_string(metadata.size()) +
                        " records but the embedding store has " + std::to_string(store.size()) +
                        " rows");
}

std::string SearchEngine::normalizeQueryText(std::string_view text) {
    std::string out;
    bool pendingSpace = false;
    for (char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            pendingSpace = !out.empty();
        } else {
            if (pendingSpace) out.push_back(' ');
            out.push_back(ch);
            pendingSpace = false;
        }
    }
    if (out.empty()) throw QueryError("query is empty or whitespace-only");
    return out;
}

void SearchEngine::checkK(int k) const {
    if (k <= 0) throw QueryError("K must be a positive integer, got " + std::to_string(k));
    if (static_cast<std::size_t>(k) > index_->size())
        throw QueryError("K=" + std::to_string(k) + " exceeds the number of indexed images (N=" +
                         std::to_string(index_->size()) + ")");
}

std::vector<float> SearchEngine::prepareVector(std::span<const float> raw) const {
    if (raw.size() != store_->dim())
        throw QueryError("query embedding dimension mismatch: got " + std::to_string(raw.size()) +
                         ", expected " + std::to_string(store_->dim()));
    for (std::size_t i = 0; i < raw.size(); ++i)
        if (!std::isfinite(raw[i]))
            throw QueryError(std::string("query embedding contains ") +
                             (std::isnan(raw[i]) ? "NaN" : "Inf") + " at element " +
                             std::to_string(i));
    std::vector<float> v(raw.begin(), raw.end());
    if (norm(v) == 0.0) throw QueryError("query embedding has zero norm");
    normalize(v);
    return v;
}

std::vector<float> SearchEngine::embedQuery(std::string_view text) const {
    const std::string clean = normalizeQueryText(text);
    if (!encoder_) throw QueryError("no text encoder configured for text queries");
    return prepareVector(encoder_->encode(clean));
}

SearchResponse SearchEngine::searchEmbedding(std::span<const float> unitQuery, int k) const {
    checkK(k);
    SearchResponse resp;
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<Candidate> hits = index_->search(unitQuery, k);
    resp.retrievalMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    resp.results.reserve(hits.size());
    int rank = 1;
    for (const Candidate& c : hits) {
        const ImageRecord& r = metadata_->at(c.id);
        resp.results.push_back({rank++, c.id, r.imageId, r.path, c.score});
    }
    return resp;
}

SearchResponse SearchEngine::search(std::string_view text, int k) const {
    checkK(k);  // fail fast before running the encoder
    const std::vector<float> q = embedQuery(text);
    return searchEmbedding(q, k);
}

}  // namespace sir
