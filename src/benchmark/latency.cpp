#include "benchmark/latency.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <set>

namespace sir {

LatencyStats summarize(std::vector<double> ms) {
    LatencyStats s;
    s.samples = ms.size();
    if (ms.empty()) return s;
    std::sort(ms.begin(), ms.end());
    const std::size_t n = ms.size();
    s.medianMs = n % 2 ? ms[n / 2] : 0.5 * (ms[n / 2 - 1] + ms[n / 2]);
    // Nearest-rank percentile.
    const auto rank = static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n)));
    s.p95Ms = ms[std::max<std::size_t>(rank, 1) - 1];
    s.meanMs = std::accumulate(ms.begin(), ms.end(), 0.0) / static_cast<double>(n);
    s.minMs = ms.front();
    s.maxMs = ms.back();
    return s;
}

std::vector<double> timeQueries(const RetrievalIndex& index, const EmbeddingStore& queries,
                                int k, int warmup, int repetitions) {
    const std::size_t nq = queries.size();
    volatile std::size_t sink = 0;  // keeps results observable so no call is elided
    for (int w = 0; w < warmup; ++w)
        sink = sink + index.search(queries.row(static_cast<std::size_t>(w) % nq), k).size();
    std::vector<std::vector<double>> perQuery(nq);
    for (int r = 0; r < repetitions; ++r) {
        for (std::size_t q = 0; q < nq; ++q) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto res = index.search(queries.row(q), k);
            const auto t1 = std::chrono::steady_clock::now();
            sink = sink + static_cast<std::size_t>(res.front().id);
            perQuery[q].push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }
    std::vector<double> out(nq);
    for (std::size_t q = 0; q < nq; ++q) out[q] = summarize(perQuery[q]).medianMs;
    return out;
}

double recallAtK(const std::vector<Candidate>& approx, const std::vector<Candidate>& exact,
                 std::size_t k) {
    std::set<int> truth;
    for (std::size_t i = 0; i < std::min(k, exact.size()); ++i) truth.insert(exact[i].id);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < std::min(k, approx.size()); ++i) hits += truth.count(approx[i].id);
    return static_cast<double>(hits) / static_cast<double>(k);
}

bool hitAtK(const std::vector<Candidate>& results, int target, std::size_t k) {
    for (std::size_t i = 0; i < std::min(k, results.size()); ++i)
        if (results[i].id == target) return true;
    return false;
}

}  // namespace sir
