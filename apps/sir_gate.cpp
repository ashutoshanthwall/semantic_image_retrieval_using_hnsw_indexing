// Verification helper for gates that compare C++ against Python/NumPy or across
// processes. Subcommands:
//   dump-rows <id>...                 exact row values (hex floats)            [G5]
//   bf-topk --n 50 --k 10 --out f     brute-force top-k for the first n queries [G7]
//   persist-save --index f --out r    build HNSW, save, record 50 query results [G10]
//   persist-check --index f --expect r   load in a fresh process, compare       [G10]
//   load-index --index f              try to load an index, report the outcome  [G10]
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include "app_common.hpp"
#include "data/errors.hpp"
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

int run(int argc, char** argv) {
    const auto args = app::parseArgs(argc, argv);
    if (args.positional.empty()) {
        std::cerr << "usage: sir_gate <dump-rows|bf-topk|persist-save|persist-check|load-index>\n";
        return 2;
    }
    const std::string cmd = args.positional[0];
    const AppConfig cfg = app::loadAppConfig(args);
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
