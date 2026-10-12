#include "benchmark/latency.hpp"

#include <chrono>
#include <cmath>

#include "algorithms/basic.hpp"
#include "algorithms/sort.hpp"
#include "containers/flat_set.hpp"

namespace sir {

LatencyStats summarize(Vector<double> ms) {
    LatencyStats s;
    s.samples = ms.size();
    if (ms.empty()) return s;
    sortRange(ms.begin(), ms.end());
    const std::size_t n = ms.size();
    s.medianMs = n % 2 ? ms[n / 2] : 0.5 * (ms[n / 2 - 1] + ms[n / 2]);
    // Nearest-rank percentile.
    const auto rank = static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n)));
    s.p95Ms = ms[maxOf<std::size_t>(rank, 1) - 1];
    double sum = 0.0;
    for (double x : ms) sum += x;
    s.meanMs = sum / static_cast<double>(n);
    s.minMs = ms.front();
    s.maxMs = ms.back();
    return s;
}

Vector<double> timeQueries(const RetrievalIndex& index, const EmbeddingStore& queries,
                                int k, int warmup, int repetitions) {
    const std::size_t nq = queries.size();
    volatile std::size_t sink = 0;  // keeps results observable so no call is elided
    for (int w = 0; w < warmup; ++w)
        sink = sink + index.search(queries.row(static_cast<std::size_t>(w) % nq), k).size();
    Vector<Vector<double>> perQuery(nq);
    for (int r = 0; r < repetitions; ++r) {
        for (std::size_t q = 0; q < nq; ++q) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto res = index.search(queries.row(q), k);
            const auto t1 = std::chrono::steady_clock::now();
            sink = sink + static_cast<std::size_t>(res.front().id);
            perQuery[q].push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }
    Vector<double> out(nq);
    for (std::size_t q = 0; q < nq; ++q) out[q] = summarize(perQuery[q]).medianMs;
    return out;
}

double recallAtK(const Vector<Candidate>& approx, const Vector<Candidate>& exact,
                 std::size_t k) {
    FlatSet<int> truth;
    for (std::size_t i = 0; i < minOf(k, exact.size()); ++i) truth.insert(exact[i].id);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < minOf(k, approx.size()); ++i) hits += truth.count(approx[i].id);
    return static_cast<double>(hits) / static_cast<double>(k);
}

bool hitAtK(const Vector<Candidate>& results, int target, std::size_t k) {
    for (std::size_t i = 0; i < minOf(k, results.size()); ++i)
        if (results[i].id == target) return true;
    return false;
}

}  // namespace sir
