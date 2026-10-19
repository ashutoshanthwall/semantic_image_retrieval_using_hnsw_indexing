// Verification helper for gates that compare C++ against Python/NumPy or across
// processes. Subcommands:
//   dump-rows <id>...                 exact row values (hex floats)            [G5]
//   bf-topk --n 50 --k 10 --out f     brute-force top-k for the first n queries [G7]
//   persist-save --index f --out r    build HNSW, save, record 50 query results [G10]
//   persist-check --index f --expect r   load in a fresh process, compare       [G10]
//   load-index --index f              try to load an index, report the outcome  [G10]
//   caption-index --n 200             caption inverted index vs a linear scan    [G12]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "app_common.hpp"
#include "data/errors.hpp"
#include "data/inverted_index.hpp"
#include "index/brute_force_index.hpp"
#include "persistence/index_manager.hpp"

using namespace sir;

namespace {

std::string hexFloat(float f) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%a", static_cast<double>(f));
    return buf;
}

std::string resultsText(const HNSWIndex& idx, const EmbeddingStore& queries, int n, int k) {
    std::ostringstream os;
    for (int q = 0; q < n; ++q)
        for (const Candidate& c : idx.search(queries.row(static_cast<std::size_t>(q)), k))
            os << q << ' ' << c.id << ' ' << hexFloat(c.score) << '\n';
    return os.str();
}

bool hasWord(const Vector<std::string>& doc, const std::string& w) {
    for (const auto& x : doc)
        if (x == w) return true;
    return false;
}

// Builds the caption inverted index, then answers n random AND and OR queries with it and
// with a linear scan over every image's words; the answers must be identical. Query words
// come from random indexed captions, so most queries have matches.
int captionIndexGate(const AppConfig& cfg, int n) {
    using Clock = std::chrono::steady_clock;
    auto msSince = [](Clock::time_point t) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    };
    const MetadataManager meta = MetadataManager::load(cfg.metadata, cfg.root);
    const int skip = cfg.evaluation.heldoutCaptionIndex;
    const auto t0 = Clock::now();
    const InvertedIndex idx = InvertedIndex::build(meta, skip);
    const double buildMs = msSince(t0);

    // Baseline data: each image's distinct words (the forward index).
    Vector<Vector<std::string>> docs(meta.size());
    for (std::size_t i = 0; i < meta.size(); ++i) {
        const auto& caps = meta.at(static_cast<int>(i)).captions;
        for (std::size_t c = 0; c < caps.size(); ++c)
            if (static_cast<int>(c) != skip)
                for (auto& w : tokenize(caps[c]))
                    if (!hasWord(docs[i], w)) docs[i].push_back(std::move(w));
    }

    std::mt19937_64 rng(cfg.evaluation.seed);
    Vector<std::string> queries;
    while (queries.size() < static_cast<std::size_t>(n)) {
        const auto& caps = meta.at(static_cast<int>(rng() % meta.size())).captions;
        const auto c = rng() % caps.size();
        if (static_cast<int>(c) == skip) continue;
        const Vector<std::string> ws = tokenize(caps[c]);
        if (ws.empty()) continue;
        std::string q;
        for (auto t = 1 + rng() % 3; t > 0; --t) q += ws[rng() % ws.size()] + ' ';
        queries.push_back(q);
    }

    int mismatches = 0;
    std::size_t hitsAll = 0, hitsAny = 0;
    double idxMs = 0, scanMs = 0;
    for (const std::string& q : queries) {
        Vector<std::string> ws = tokenize(q);
        auto t = Clock::now();
        const Vector<int> all = idx.matchAll(q), any = idx.matchAny(q);
        idxMs += msSince(t);
        t = Clock::now();
        Vector<int> wantAll, wantAny;
        for (std::size_t d = 0; d < docs.size(); ++d) {
            int found = 0;
            for (const auto& w : ws) found += hasWord(docs[d], w);
            if (found == static_cast<int>(ws.size())) wantAll.push_back(static_cast<int>(d));
            if (found > 0) wantAny.push_back(static_cast<int>(d));
        }
        scanMs += msSince(t);
        mismatches += !(all == wantAll) + !(any == wantAny);
        hitsAll += all.size();
        hitsAny += any.size();
    }

    const auto& table = idx.terms();
    std::cout << std::fixed << std::setprecision(3) << "images=" << meta.size()
              << " skipped_caption=" << skip << " vocabulary=" << idx.vocabularySize()
              << " postings=" << idx.postingCount() << " build_ms=" << buildMs << '\n'
              << "hash table: capacity=" << table.capacity()
              << " load_factor=" << table.loadFactor()
              << " avg_probe_length=" << table.averageProbeLength() << '\n'
              << "queries=" << n << " (1-3 words) mean_hits AND=" << double(hitsAll) / n
              << " OR=" << double(hitsAny) / n << '\n'
              << "AND+OR per query: index_us=" << 1000 * idxMs / n
              << " linear_scan_us=" << 1000 * scanMs / n
              << " speedup=" << std::setprecision(1) << scanMs / idxMs << "x\n"
              << "result mismatches vs linear scan: " << mismatches << " -> "
              << (mismatches == 0 ? "PASS" : "FAIL") << '\n';
    return mismatches == 0 ? 0 : 1;
}

int run(int argc, char** argv) {
    const auto args = app::parseArgs(argc, argv);
    if (args.positional.empty()) {
        std::cerr << "usage: sir_gate <dump-rows|bf-topk|persist-save|persist-check|load-index|"
                     "caption-index>\n";
        return 2;
    }
    const std::string cmd = args.positional[0];
    const AppConfig cfg = app::loadAppConfig(args);
    if (cmd == "caption-index") return captionIndexGate(cfg, args.getInt("n", 200));
    const EmbeddingStore store = app::loadImageStore(cfg);

    if (cmd == "dump-rows") {
        std::cout << "rows=" << store.size() << " dim=" << store.dim() << '\n';
        for (std::size_t p = 1; p < args.positional.size(); ++p) {
            const auto id = static_cast<std::size_t>(std::stoul(args.positional[p]));
            if (id >= store.size()) throw std::out_of_range("row id out of range");
            std::cout << id;
            for (float x : store.row(id)) std::cout << ' ' << hexFloat(x);
            std::cout << '\n';
        }
        return 0;
    }
    const EmbeddingStore queries = app::loadQueryStore(cfg);
    const int n = args.getInt("n", 50), k = args.getInt("k", 10);

    if (cmd == "bf-topk") {
        BruteForceIndex bf(store);
        bf.insertAll();
        std::ofstream out(args.get("out", "bf_topk.txt"));
        for (int q = 0; q < n; ++q)
            for (const Candidate& c : bf.search(queries.row(static_cast<std::size_t>(q)), k))
                out << q << ' ' << c.id << ' ' << hexFloat(c.score) << '\n';
        std::cout << "wrote brute-force top-" << k << " for " << n << " queries\n";
        return 0;
    }
    if (cmd == "persist-save") {
        BuildStats st;
        const HNSWIndex idx = IndexManager::build(store, cfg.hnsw, &st);
        IndexManager::save(idx, cfg.modelName, args.get("index", ""));
        std::ofstream(args.get("out", "")) << resultsText(idx, queries, n, k);
        std::cout << "built in " << st.seconds << " s, saved, recorded " << n
                  << " query results\n";
        return 0;
    }
    if (cmd == "persist-check") {
        const HNSWIndex idx = IndexManager::load(args.get("index", ""), store, cfg.modelName);
        std::ifstream in(args.get("expect", ""));
        std::stringstream want;
        want << in.rdbuf();
        const std::string got = resultsText(idx, queries, n, k);
        std::istringstream a(want.str()), b(got);
        std::string la, lb;
        int lines = 0, diff = 0;
        while (std::getline(a, la)) {
            std::getline(b, lb);
            ++lines;
            diff += (la != lb);
        }
        std::cout << "compared " << lines << " (query, rank) results: " << diff
                  << " differences -> " << (diff == 0 && lines == n * k ? "IDENTICAL" : "MISMATCH")
                  << '\n';
        return diff == 0 && lines == n * k ? 0 : 1;
    }
    if (cmd == "load-index") {
        try {
            const HNSWIndex idx = IndexManager::load(args.get("index", ""), store, cfg.modelName);
            std::cout << "loaded OK: " << idx.size() << " nodes\n";
            return 0;
        } catch (const PersistenceError& e) {
            std::cout << "load rejected: " << e.what() << '\n';
            return 3;
        }
    }
    std::cerr << "unknown subcommand " << cmd << '\n';
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
