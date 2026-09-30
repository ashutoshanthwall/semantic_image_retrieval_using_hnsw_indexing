// Builds the HNSW index over the full embedding store, validates the graph, saves it, and
// reports Recall@10 against brute force for each efSearch in the sweep (gate G9).
//
//   sir_build_index [--config f] [--out path] [--M 16] [--ef-construction 200]
//                   [--select-heuristic heuristic|naive] [--no-recall]
#include <iomanip>
#include <iostream>
#include <set>

#include "app_common.hpp"
#include "index/brute_force_index.hpp"
#include "persistence/index_manager.hpp"

using namespace sir;

int main(int argc, char** argv) {
    try {
        const auto args = app::parseArgs(argc, argv, {"no-recall"});
        const AppConfig cfg = app::loadAppConfig(args);
        HnswParams p = cfg.hnsw;
        p.M = args.getInt("M", p.M);
        p.efConstruction = args.getInt("ef-construction", p.efConstruction);
        if (args.has("select-heuristic")) {
            const std::string mode = args.get("select-heuristic", "");
            if (mode != "heuristic" && mode != "naive")
                throw std::invalid_argument("--select-heuristic must be 'heuristic' or 'naive'");
            p.selectHeuristic = mode == "heuristic";
        }
        if (args.has("keep-pruned")) p.keepPrunedConnections = args.get("keep-pruned", "1") == "1";
        if (args.has("protect-in-edge")) p.protectLastInEdge = args.get("protect-in-edge", "1") == "1";
        const std::filesystem::path out = args.get("out", cfg.index.string());

        const EmbeddingStore store = app::loadImageStore(cfg);
        std::cout << "embeddings: " << store.size() << " x " << store.dim()
                  << " (sha256 " << store.sourceSha256().substr(0, 16) << "...)\n"
                  << "HNSW: M=" << p.M << " efConstruction=" << p.efConstruction
                  << " seed=" << p.seed
                  << " selection=" << (p.selectHeuristic ? "heuristic" : "naive")
                  << " keepPruned=" << p.keepPrunedConnections
                  << " protectLastInEdge=" << p.protectLastInEdge << '\n';
        BuildStats st;
        HNSWIndex idx = IndexManager::rebuild(store, p, cfg.modelName, out, &st);
        const IndexValidation v = IndexManager::validate(idx);
        std::cout << std::fixed << std::setprecision(2) << "built in " << st.seconds
                  << " s; saved to " << out.string() << " ("
                  << std::filesystem::file_size(out) / 1024 << " KiB)\n"
                  << "graph: nodes=" << v.nodes << " maxLevel=" << v.maxLevel
                  << " meanDegree(L0)=" << v.meanDegreeLayer0
                  << " reachable(L0)=" << v.reachableLayer0 << "/" << v.nodes << '\n'
                  << "nodes per level:";
        for (std::size_t l = 0; l < v.nodesPerLevel.size(); ++l)
            std::cout << " L" << l << "=" << v.nodesPerLevel[l];
        std::cout << "\nlayer-0 nodes with in-degree 0 (excluding entry): " << v.inDegreeZeroLayer0;
        if (!v.unreachableLayer0.empty()) {
            // Diagnose each unreachable node: its exact nearest neighbours.
            BruteForceIndex probe(store);
            probe.insertAll();
            for (int u : v.unreachableLayer0) {
                std::cout << "\n  unreachable id=" << u << " level=" << idx.levelOf(u)
                          << " outDegree=" << idx.neighbors(u, 0).size() << " exact NN:";
                for (const Candidate& c : probe.search(store.row(static_cast<std::size_t>(u)), 3))
                    if (c.id != u)
                        std::cout << ' ' << c.id << "(sim " << std::setprecision(5) << c.score
                                  << std::setprecision(2) << ")";
            }
        }
        std::cout << "\nmemoryBytes(): HNSW graph " << idx.memoryBytes() << ", embeddings "
                  << store.memoryBytes() << '\n';
        if (args.has("no-recall")) return 0;

        const EmbeddingStore queries = app::loadQueryStore(cfg);
        BruteForceIndex bf(store);
        bf.insertAll();
        std::vector<std::set<int>> truth(queries.size());
        for (std::size_t q = 0; q < queries.size(); ++q)
            for (const Candidate& c : bf.search(queries.row(q), 10)) truth[q].insert(c.id);
        std::cout << "Recall@10 vs brute force over " << queries.size() << " queries:\n";
        for (int ef : cfg.evaluation.efSearchSweep) {
            idx.setEfSearch(ef);
            std::size_t hits = 0;
            idx.resetDistanceComputations();
            for (std::size_t q = 0; q < queries.size(); ++q)
                for (const Candidate& c : idx.search(queries.row(q), 10)) hits += truth[q].count(c.id);
            std::cout << "  efSearch=" << std::setw(4) << ef << "  Recall@10="
                      << std::setprecision(4)
                      << static_cast<double>(hits) / (10.0 * static_cast<double>(queries.size()))
                      << "  mean distance evaluations/query="
                      << std::setprecision(0)
                      << static_cast<double>(idx.distanceComputations()) /
                             static_cast<double>(queries.size())
                      << std::setprecision(2) << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
