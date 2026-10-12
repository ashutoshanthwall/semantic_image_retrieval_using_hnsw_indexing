#pragma once
// Timing protocol: `warmup` queries are run and discarded, then `repetitions` passes over
// all queries (repetition-outer, so consecutive calls never repeat the same query and hit
// an artificially warm cache). Each query's latency is the median of its repetitions;
// statistics are then taken across queries. Only RetrievalIndex::search is timed
// (std::chrono::steady_clock).
#include "containers/vector.hpp"
#include "embedding/embedding_store.hpp"
#include "index/retrieval_index.hpp"

namespace sir {

struct LatencyStats {
    double medianMs = 0, p95Ms = 0, meanMs = 0, minMs = 0, maxMs = 0;
    std::size_t samples = 0;
};

LatencyStats summarize(Vector<double> ms);

// Returns per-query latency in milliseconds (median over repetitions).
Vector<double> timeQueries(const RetrievalIndex& index, const EmbeddingStore& queries,
                                int k, int warmup, int repetitions);

// |approx ∩ exact| / k using the first k entries of each list.
double recallAtK(const Vector<Candidate>& approx, const Vector<Candidate>& exact,
                 std::size_t k);
// 1 if `target` is among the first k results.
bool hitAtK(const Vector<Candidate>& results, int target, std::size_t k);

}  // namespace sir
