// Interactive natural-language image search: brute force vs HNSW side by side.
//
//   sir_cli [--config f] [--k 5] [--ef N] [--encoder-log file] [query words...]
//
// With a query on the command line it answers once; otherwise it reads queries from stdin
// until EOF or an empty line. Text is embedded by the persistent Python CLIP helper; all
// retrieval, ranking and metadata lookup is C++.
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <set>

#include "app_common.hpp"
#include "benchmark/latency.hpp"
#include "data/errors.hpp"
#include "embedding/text_encoder_process.hpp"
#include "index/brute_force_index.hpp"
#include "persistence/index_manager.hpp"
#include "search/search_engine.hpp"

using namespace sir;

namespace {

constexpr int kTimingRuns = 5;  // a single query is timed several times; median reported

double medianRetrievalMs(const SearchEngine& e, std::span<const float> q, int k,
                         SearchResponse& out) {
    std::vector<double> ms;
    for (int i = 0; i < kTimingRuns; ++i) {
        out = e.searchEmbedding(q, k);
        ms.push_back(out.retrievalMs);
    }
    return summarize(ms).medianMs;
}

void printResults(const std::string& title, const SearchResponse& r, double ms) {
    std::cout << "\n" << title << "  (retrieval " << std::fixed << std::setprecision(3) << ms
              << " ms)\n";
    for (const SearchResult& s : r.results)
        std::cout << "  " << s.rank << ". " << std::setprecision(4) << s.score << "  "
                  << s.imageId << "  " << s.path.string()
                  << (std::filesystem::exists(s.path) ? "" : "  [MISSING ON DISK]") << "\n";
}

void answer(const std::string& query, int k, const SearchEngine& bfEngine,
            const SearchEngine& hnswEngine) {
    bfEngine.checkK(k);
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<float> q = bfEngine.embedQuery(query);
    const double encMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    SearchResponse bf, hn;
    const double bfMs = medianRetrievalMs(bfEngine, q, k, bf);
    const double hnMs = medianRetrievalMs(hnswEngine, q, k, hn);
    std::set<int> truth;
    for (const auto& s : bf.results) truth.insert(s.internalId);
    std::size_t overlap = 0;
    for (const auto& s : hn.results) overlap += truth.count(s.internalId);
    std::cout << "\nquery: \"" << SearchEngine::normalizeQueryText(query) << "\"   (text "
              << "embedding " << std::fixed << std::setprecision(1) << encMs
              << " ms, excluded from retrieval latency)";
    printResults("Brute force (exact) top-" + std::to_string(k), bf, bfMs);
    printResults("HNSW top-" + std::to_string(k), hn, hnMs);
    std::cout << "\nRecall@" << k << " (HNSW vs brute force): " << std::setprecision(2)
              << static_cast<double>(overlap) / k << "   speedup: " << std::setprecision(1)
              << bfMs / hnMs << "x   (median of " << kTimingRuns << " runs each)\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = app::parseArgs(argc, argv);
        const AppConfig cfg = app::loadAppConfig(args);
        const int k = args.getInt("k", 5);
        std::cerr << "loading metadata, embeddings and index...\n";
        const MetadataManager meta = MetadataManager::load(cfg.metadata, cfg.root);
        const EmbeddingStore store = app::loadImageStore(cfg);
        BruteForceIndex bf(store);
        bf.insertAll();
        HNSWIndex hnsw = IndexManager::load(args.get("index", cfg.index.string()), store,
                                            cfg.modelName);
        hnsw.setEfSearch(args.getInt("ef", cfg.hnsw.efSearch));
        std::cerr << "starting text encoder...\n";
        TextEncoderProcess encoder(cfg.python, cfg.textEncoderServer, cfg.configFile, cfg.dim,
                                   args.get("encoder-log", "/dev/null"));
        const SearchEngine bfEngine(meta, store, bf, &encoder);
        const SearchEngine hnswEngine(meta, store, hnsw, &encoder);
        std::cerr << "ready: " << store.size() << " images, HNSW M=" << hnsw.params().M
                  << " efSearch=" << hnsw.params().efSearch << "\n";

        if (!args.positional.empty()) {
            std::string q;
            for (const auto& w : args.positional) q += (q.empty() ? "" : " ") + w;
            answer(q, k, bfEngine, hnswEngine);
            return 0;
        }
        std::string line;
        while (std::cout << "\nsearch> " << std::flush, std::getline(std::cin, line)) {
            if (line.empty()) break;
            try {
                answer(line, k, bfEngine, hnswEngine);
            } catch (const QueryError& e) {
                std::cout << "invalid query: " << e.what() << "\n";
            }
        }
        return 0;
    } catch (const QueryError& e) {
        std::cerr << "invalid query: " << e.what() << '\n';
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
