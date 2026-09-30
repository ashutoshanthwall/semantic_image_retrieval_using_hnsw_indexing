#pragma once
// Runs one text query through both backends the same way for every front end (CLI and
// web UI): embed once, time each backend's retrieval `timingRuns` times and keep the
// median, then compute the overlap of the two top-K lists.
#include <chrono>
#include <set>
#include <string>
#include <vector>

#include "benchmark/latency.hpp"
#include "search/search_engine.hpp"

namespace sir::app {

struct Comparison {
    std::string query;  // normalised text actually sent to the encoder
    int k = 0;
    double encodeMs = 0.0;
    SearchResponse exact, approx;
    double exactMs = 0.0, approxMs = 0.0;  // median over timingRuns
    double recall = 0.0;                   // |approx ∩ exact| / k
    double speedup = 0.0;
    std::set<int> exactIds, approxIds;
};

inline double medianRetrievalMs(const SearchEngine& e, std::span<const float> q, int k,
                                int runs, SearchResponse& out) {
    std::vector<double> ms;
    for (int i = 0; i < runs; ++i) {
        out = e.searchEmbedding(q, k);
        ms.push_back(out.retrievalMs);
    }
    return summarize(ms).medianMs;
}

inline Comparison compareBackends(std::string_view text, int k, const SearchEngine& exact,
                                  const SearchEngine& approx, int timingRuns = 5) {
    exact.checkK(k);
    Comparison c;
    c.query = SearchEngine::normalizeQueryText(text);
    c.k = k;
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> q = exact.embedQuery(c.query);
    c.encodeMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    c.exactMs = medianRetrievalMs(exact, q, k, timingRuns, c.exact);
    c.approxMs = medianRetrievalMs(approx, q, k, timingRuns, c.approx);
    for (const auto& r : c.exact.results) c.exactIds.insert(r.internalId);
    std::size_t overlap = 0;
    for (const auto& r : c.approx.results) {
        c.approxIds.insert(r.internalId);
        overlap += c.exactIds.count(r.internalId);
    }
    c.recall = static_cast<double>(overlap) / k;
    c.speedup = c.exactMs / c.approxMs;
    return c;
}

}  // namespace sir::app
