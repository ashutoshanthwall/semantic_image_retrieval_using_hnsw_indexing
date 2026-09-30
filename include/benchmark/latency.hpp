#pragma once
// Timing protocol: `warmup` queries are run and discarded, then `repetitions` passes over
// all queries (repetition-outer, so consecutive calls never repeat the same query and hit
// an artificially warm cache). Each query's latency is the median of its repetitions;
// statistics are then taken across queries. Only RetrievalIndex::search is timed
// (std::chrono::steady_clock).
#include <vector>

#include "embedding/embedding_store.hpp"
#include "index/retrieval_index.hpp"

namespace sir {

struct LatencyStats {
    double medianMs = 0, p95Ms = 0, meanMs = 0, minMs = 0, maxMs = 0;
    std::size_t samples = 0;
};

LatencyStats summarize(std::vector<double> ms);

// Returns per-query latency in milliseconds (median over repetitions).
std::vector<double> timeQueries(const RetrievalIndex& index, const EmbeddingStore& queries,
                                int k, int warmup, int repetitions);

// |approx ∩ exact| / k using the first k entries of each list.
double recallAtK(const std::vector<Candidate>& approx, const std::vector<Candidate>& exact,
                 std::size_t k);
// 1 if `target` is among the first k results.
bool hitAtK(const std::vector<Candidate>& results, int target, std::size_t k);

}  // namespace sir
