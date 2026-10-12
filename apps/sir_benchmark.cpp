// Exact vs approximate retrieval benchmark (single-threaded, precomputed query embeddings).
//
//   sir_benchmark [--config f] [--out-dir d] [--measure-encoder] [--pin-cpu N]
//
// --pin-cpu pins this process to one CPU (sched_setaffinity) so every backend runs on the
// same core type; on hybrid CPUs, P- and E-cores differ greatly in speed. The optional
// text-encoder helper is spawned before pinning so it does not inherit the affinity.
//
// Measures, for brute force and for HNSW at every efSearch in the sweep (M and
// efConstruction from the config) with three neighbour-selection variants:
//   latency (median / p95 / mean of per-query medians), speedup, Recall@5/@10 against
//   brute force, distance evaluations per query, build time, memory (VmHWM and
//   memoryBytes()), and the absolute text->image quality track R@1/5/10 on held-out
//   captions. Writes CSV + JSON + a markdown summary. Every number is measured here.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <memory>
#include <sstream>

#include <sched.h>

#include "app_common.hpp"
#include "benchmark/latency.hpp"
#include "algorithms/basic.hpp"
#include "benchmark/system_info.hpp"
#include "data/errors.hpp"
#include "embedding/text_encoder_process.hpp"
#include "index/brute_force_index.hpp"
#include "persistence/index_manager.hpp"

using namespace sir;
using nlohmann::json;

namespace {

struct Variant {
    std::string name;
    HnswParams params;
};

struct Row {
    std::string backend, variant;
    int efSearch = 0;
    LatencyStats lat;
    double speedup = 1.0, recall5 = 1.0, recall10 = 1.0, perfect10 = 1.0;
    double distEvals = 0.0, buildSeconds = 0.0;
    std::size_t structBytes = 0, peakRss = 0;
    std::size_t reachable = 0;
    double r1 = 0, r5 = 0, r10 = 0;  // quality track
};

Vector<int> loadGroundTruth(const std::filesystem::path& csv) {
    std::ifstream in(csv);
    if (!in) throw DataError("cannot open " + csv.string());
    std::string line;
    std::getline(in, line);  // header
    Vector<int> gt;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string idx, kind, imageId, internalId;
        std::getline(ss, idx, ',');
        std::getline(ss, kind, ',');
        std::getline(ss, imageId, ',');
        std::getline(ss, internalId, ',');
        gt.push_back(std::stoi(internalId));
    }
    return gt;
}

Vector<std::string> loadQueryTexts(const std::filesystem::path& txt) {
    std::ifstream in(txt);
    Vector<std::string> q;
    for (std::string line; std::getline(in, line);)
        if (!line.empty()) q.push_back(line);
    return q;
}

void qualityInto(Row& row, const Vector<Vector<Candidate>>& res, const Vector<int>& gt) {
    std::size_t n = 0, h1 = 0, h5 = 0, h10 = 0;
    for (std::size_t q = 0; q < res.size(); ++q) {
        if (gt[q] < 0) continue;  // hand-written queries have no ground truth
        ++n;
        h1 += hitAtK(res[q], gt[q], 1);
        h5 += hitAtK(res[q], gt[q], 5);
        h10 += hitAtK(res[q], gt[q], 10);
    }
    row.r1 = static_cast<double>(h1) / static_cast<double>(n);
    row.r5 = static_cast<double>(h5) / static_cast<double>(n);
    row.r10 = static_cast<double>(h10) / static_cast<double>(n);
}

std::string fmt(double v, int p) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(p) << v;
    return os.str();
}

std::string mib(std::size_t b) { return fmt(static_cast<double>(b) / (1024.0 * 1024.0), 2); }

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = app::parseArgs(argc, argv, {"measure-encoder"});
        const AppConfig cfg = app::loadAppConfig(args);
        const std::filesystem::path outDir = args.get("out-dir", cfg.benchmarkResults.string());
        std::filesystem::create_directories(outDir);
        std::unique_ptr<TextEncoderProcess> encoder;
        if (args.has("measure-encoder"))
            encoder = std::make_unique<TextEncoderProcess>(cfg.python, cfg.textEncoderServer,
                                                           cfg.configFile, cfg.dim);
        if (args.has("pin-cpu")) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(args.getInt("pin-cpu", 0), &set);
            if (::sched_setaffinity(0, sizeof set, &set) != 0)
                throw std::runtime_error("sched_setaffinity failed for --pin-cpu");
        }
        const SystemInfo sys = collectSystemInfo(cfg.root);
        const int K = 10, warm = cfg.evaluation.warmupQueries, reps = cfg.evaluation.repetitions;

        std::cout << "CPU: " << sys.cpuModel << " | logical CPUs " << sys.logicalCpus
                  << " | physical cores " << sys.physicalCores << " | RAM " << mib(sys.ramBytes)
                  << " MiB\nOS: " << sys.os << " (" << sys.kernel << ")\ncompiler: "
                  << sys.compiler << " | " << sys.buildType << " " << sys.cxxFlags
                  << "\ngit: " << sys.gitCommit << " | governor " << sys.cpuGovernor
                  << " | affinity CPUs " << sys.cpuAffinity << " | loadavg " << sys.loadAverage
                  << "\n";

        const EmbeddingStore store = app::loadImageStore(cfg);
        const EmbeddingStore queries = app::loadQueryStore(cfg);
        const Vector<int> gt = loadGroundTruth(cfg.queryGroundTruth);
        if (gt.size() != queries.size())
            throw DataError("query ground truth has " + std::to_string(gt.size()) +
                            " rows but there are " + std::to_string(queries.size()) + " queries");
        const std::size_t rssLoaded = peakRssBytes();
        std::cout << "images " << store.size() << " x " << store.dim() << ", queries "
                  << queries.size() << ", warm-up " << warm << ", repetitions " << reps
                  << ", K " << K << "\n";

        // ---------------------------------------------------------------- brute force
        Vector<Row> rows;
        BruteForceIndex bf(store);
        const auto tb = std::chrono::steady_clock::now();
        bf.insertAll();
        const double bfBuild =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count();
        Vector<Vector<Candidate>> exact(queries.size());
        for (std::size_t q = 0; q < queries.size(); ++q) exact[q] = bf.search(queries.row(q), K);
        Row bfRow;
        bfRow.backend = "brute_force";
        bfRow.variant = "exact";
        bfRow.lat = summarize(timeQueries(bf, queries, K, warm, reps));
        bfRow.distEvals = static_cast<double>(store.size());
        bfRow.buildSeconds = bfBuild;
        bfRow.structBytes = bf.memoryBytes();
        bfRow.peakRss = peakRssBytes();
        bfRow.reachable = store.size();
        qualityInto(bfRow, exact, gt);
        rows.push_back(bfRow);
        std::cout << "brute force: median " << fmt(bfRow.lat.medianMs, 3) << " ms, p95 "
                  << fmt(bfRow.lat.p95Ms, 3) << " ms, mean " << fmt(bfRow.lat.meanMs, 3)
                  << " ms | R@1 " << fmt(bfRow.r1, 4) << "\n";

        // ---------------------------------------------------------------- HNSW
        Vector<Variant> variants;
        HnswParams base = cfg.hnsw;
        base.selectHeuristic = true;
        base.keepPrunedConnections = true;
        variants.push_back({"heuristic+keepPruned", base});
        HnswParams noKeep = base;
        noKeep.keepPrunedConnections = false;
        variants.push_back({"heuristic", noKeep});
        HnswParams naive = base;
        naive.selectHeuristic = false;
        naive.keepPrunedConnections = false;
        variants.push_back({"naive", naive});

        for (const Variant& v : variants) {
            BuildStats bs;
            HNSWIndex h = IndexManager::build(store, v.params, &bs);
            const IndexValidation val = IndexManager::validate(h);
            const std::size_t rssBuilt = peakRssBytes();
            std::cout << "HNSW " << v.name << ": built in " << fmt(bs.seconds, 2)
                      << " s, reachable " << val.reachableLayer0 << "/" << val.nodes
                      << ", mean L0 degree " << fmt(val.meanDegreeLayer0, 2) << "\n";
            for (int ef : cfg.evaluation.efSearchSweep) {
                h.setEfSearch(ef);
                Row r;
                r.backend = "hnsw";
                r.variant = v.name;
                r.efSearch = ef;
                Vector<Vector<Candidate>> approx(queries.size());
                h.resetDistanceComputations();
                double s5 = 0, s10 = 0, perfect = 0;
                for (std::size_t q = 0; q < queries.size(); ++q) {
                    approx[q] = h.search(queries.row(q), K);
                    s5 += recallAtK(approx[q], exact[q], 5);
                    const double r10 = recallAtK(approx[q], exact[q], 10);
                    s10 += r10;
                    perfect += r10 == 1.0;
                }
                const auto nq = static_cast<double>(queries.size());
                r.distEvals = static_cast<double>(h.distanceComputations()) / nq;
                r.recall5 = s5 / nq;
                r.recall10 = s10 / nq;
                r.perfect10 = perfect / nq;
                qualityInto(r, approx, gt);
                r.lat = summarize(timeQueries(h, queries, K, warm, reps));
                r.speedup = bfRow.lat.medianMs / r.lat.medianMs;
                r.buildSeconds = bs.seconds;
                r.structBytes = h.memoryBytes();
                r.peakRss = rssBuilt;
                r.reachable = val.reachableLayer0;
                rows.push_back(r);
                std::cout << "  ef=" << std::setw(3) << ef << "  median " << fmt(r.lat.medianMs, 4)
                          << " ms  p95 " << fmt(r.lat.p95Ms, 4) << "  speedup "
                          << fmt(r.speedup, 1) << "x  R@5 " << fmt(r.recall5, 4) << "  R@10 "
                          << fmt(r.recall10, 4) << "  dist/q " << fmt(r.distEvals, 0)
                          << "  quality R@1 " << fmt(r.r1, 4) << "\n";
            }
        }

        // ---------------------------------------------------------- text encoder latency
        LatencyStats enc;
        if (encoder) {
            TextEncoderProcess& te = *encoder;
            const auto texts = loadQueryTexts(cfg.queries);
            for (int w = 0; w < 5; ++w) te.encode(texts[static_cast<std::size_t>(w)]);
            Vector<double> ms;
            for (std::size_t i = 0; i < minOf<std::size_t>(100, texts.size()); ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                te.encode(texts[i]);
                ms.push_back(std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count());
            }
            enc = summarize(ms);
            std::cout << "text encoding (Python helper, CPU, incl. IPC; " << enc.samples
                      << " queries): median " << fmt(enc.medianMs, 2) << " ms, p95 "
                      << fmt(enc.p95Ms, 2) << " ms\n";
        }

        // ---------------------------------------------------------------- outputs
        {
            std::ofstream c(outDir / "latency_recall.csv");
            c << "backend,variant,M,ef_construction,ef_search,median_ms,p95_ms,mean_ms,min_ms,"
                 "max_ms,speedup_vs_bf_median,recall_at_5,recall_at_10,frac_queries_recall10_eq_1,"
                 "dist_evals_per_query,build_seconds,memory_bytes_structure,peak_rss_bytes,"
                 "reachable_layer0\n";
            for (const Row& r : rows)
                c << r.backend << ',' << r.variant << ',' << (r.backend == "hnsw" ? cfg.hnsw.M : 0)
                  << ',' << (r.backend == "hnsw" ? cfg.hnsw.efConstruction : 0) << ','
                  << r.efSearch << ',' << fmt(r.lat.medianMs, 5) << ',' << fmt(r.lat.p95Ms, 5)
                  << ',' << fmt(r.lat.meanMs, 5) << ',' << fmt(r.lat.minMs, 5) << ','
                  << fmt(r.lat.maxMs, 5) << ',' << fmt(r.speedup, 3) << ',' << fmt(r.recall5, 5)
                  << ',' << fmt(r.recall10, 5) << ',' << fmt(r.perfect10, 5) << ','
                  << fmt(r.distEvals, 1) << ',' << fmt(r.buildSeconds, 3) << ',' << r.structBytes
                  << ',' << r.peakRss << ',' << r.reachable << '\n';
            std::ofstream qcsv(outDir / "quality.csv");
            qcsv << "backend,variant,ef_search,n_queries,R_at_1,R_at_5,R_at_10\n";
            std::size_t nHeld = 0;
            for (int g : gt) nHeld += g >= 0;
            for (const Row& r : rows)
                qcsv << r.backend << ',' << r.variant << ',' << r.efSearch << ',' << nHeld << ','
                     << fmt(r.r1, 5) << ',' << fmt(r.r5, 5) << ',' << fmt(r.r10, 5) << '\n';
        }
        json sj = {{"cpu_model", sys.cpuModel},       {"logical_cpus", sys.logicalCpus},
                   {"physical_cores", sys.physicalCores}, {"ram_bytes", sys.ramBytes},
                   {"os", sys.os},                    {"kernel", sys.kernel},
                   {"compiler", sys.compiler},        {"build_type", sys.buildType},
                   {"cxx_flags", sys.cxxFlags},       {"git_commit", sys.gitCommit},
                   {"cpu_governor", sys.cpuGovernor}, {"cpu_affinity", sys.cpuAffinity},
                   {"loadavg_at_start", sys.loadAverage}, {"timestamp_utc", sys.timestampUtc},
                   {"hnsw_seed", cfg.hnsw.seed},      {"M", cfg.hnsw.M},
                   {"ef_construction", cfg.hnsw.efConstruction},
                   {"queries", queries.size()},       {"warmup_queries", warm},
                   {"repetitions", reps},             {"k", K},
                   {"images", store.size()},          {"dim", store.dim()},
                   {"embedding_store_bytes", store.memoryBytes()},
                   {"peak_rss_after_loading_bytes", rssLoaded},
                   {"final_peak_rss_bytes", peakRssBytes()}};
        if (enc.samples)
            sj["text_encoder_ms"] = {{"median", enc.medianMs}, {"p95", enc.p95Ms},
                                     {"mean", enc.meanMs}, {"samples", enc.samples}};
        std::ofstream(outDir / "system.json") << sj.dump(2) << '\n';

        std::ofstream md(outDir / "summary.md");
        md << "# Benchmark results\n\nGenerated by `sir_benchmark` at " << sys.timestampUtc
           << " (git " << sys.gitCommit << "). All numbers measured in this run.\n\n"
           << "| Environment | |\n|---|---|\n| CPU | " << sys.cpuModel << " (" << sys.logicalCpus
           << " logical CPUs, " << sys.physicalCores << " physical cores) |\n| RAM | "
           << mib(sys.ramBytes) << " MiB |\n| OS | " << sys.os << ", " << sys.kernel
           << " |\n| Compiler | " << sys.compiler << " |\n| Flags | " << sys.buildType << " `"
           << sys.cxxFlags << "` |\n| CPU governor / affinity | " << sys.cpuGovernor
           << " / CPUs " << sys.cpuAffinity << " |\n| Load average at start | "
           << sys.loadAverage << " |\n\n"
           << "Protocol: " << store.size() << " images x " << store.dim() << " dims; "
           << queries.size() << " queries (5 hand-written + held-out captions); " << warm
           << " warm-up queries discarded; " << reps
           << " timed repetitions per query (repetition-outer); per-query latency = median of "
              "its repetitions; statistics across queries. Single thread, scalar distance "
              "kernel shared by both backends. K = " << K << ". HNSW M = " << cfg.hnsw.M
           << ", efConstruction = " << cfg.hnsw.efConstruction << ", seed = " << cfg.hnsw.seed
           << ", last-in-edge protection on.\n\n"
           << "## Latency and recall vs brute force\n\n"
           << "| Backend | Selection | efSearch | Median ms | p95 ms | Mean ms | Speedup | "
              "Recall@5 | Recall@10 | Queries w/ R@10=1 | Dist evals/query | Build s |\n"
           << "|---|---|---|---|---|---|---|---|---|---|---|---|\n";
        for (const Row& r : rows)
            md << "| " << r.backend << " | " << r.variant << " | "
               << (r.efSearch ? std::to_string(r.efSearch) : "-") << " | "
               << fmt(r.lat.medianMs, 4) << " | " << fmt(r.lat.p95Ms, 4) << " | "
               << fmt(r.lat.meanMs, 4) << " | " << fmt(r.speedup, 1) << "x | "
               << fmt(r.recall5, 4) << " | " << fmt(r.recall10, 4) << " | "
               << fmt(100.0 * r.perfect10, 1) << "% | " << fmt(r.distEvals, 0) << " | "
               << fmt(r.buildSeconds, 2) << " |\n";
        md << "\n## Neighbour selection at efSearch = 80\n\n| Selection | Recall@10 | Median ms | "
              "Dist evals/query | Build s | Graph bytes |\n|---|---|---|---|---|---|\n";
        for (const Row& r : rows)
            if (r.efSearch == 80)
                md << "| " << r.variant << " | " << fmt(r.recall10, 4) << " | "
                   << fmt(r.lat.medianMs, 4) << " | " << fmt(r.distEvals, 0) << " | "
                   << fmt(r.buildSeconds, 2) << " | " << r.structBytes << " |\n";
        md << "\n## Memory\n\n| Structure | memoryBytes() | MiB |\n|---|---|---|\n"
           << "| EmbeddingStore (shared by both backends) | " << store.memoryBytes() << " | "
           << mib(store.memoryBytes()) << " |\n| BruteForceIndex (id list) | " << bfRow.structBytes
           << " | " << mib(bfRow.structBytes) << " |\n";
        for (const Row& r : rows)
            if (r.efSearch == cfg.evaluation.efSearchSweep.front())
                md << "| HNSWIndex graph (" << r.variant << ") | " << r.structBytes << " | "
                   << mib(r.structBytes) << " |\n";
        md << "\nPeak RSS (VmHWM, whole process, monotone): after loading embeddings + queries "
           << mib(rssLoaded) << " MiB; after brute-force timing " << mib(bfRow.peakRss)
           << " MiB; at end " << mib(peakRssBytes())
           << " MiB. VmHWM includes the C++ runtime, both embedding files, ground-truth result "
              "lists and allocator slack, so it is larger than the sum of memoryBytes().\n"
           << "\n## Absolute retrieval quality (held-out caption #1 -> its image)\n\n"
           << "| Backend | Selection | efSearch | R@1 | R@5 | R@10 |\n|---|---|---|---|---|---|\n";
        for (const Row& r : rows)
            md << "| " << r.backend << " | " << r.variant << " | "
               << (r.efSearch ? std::to_string(r.efSearch) : "-") << " | " << fmt(r.r1, 4)
               << " | " << fmt(r.r5, 4) << " | " << fmt(r.r10, 4) << " |\n";
        if (enc.samples)
            md << "\n## Query embedding latency (excluded from retrieval latency)\n\nPython CLIP "
                  "text-encoder helper on CPU, including socket IPC, " << enc.samples
               << " queries: median " << fmt(enc.medianMs, 2) << " ms, p95 "
               << fmt(enc.p95Ms, 2) << " ms, mean " << fmt(enc.meanMs, 2) << " ms.\n";
        std::cout << "wrote " << (outDir / "summary.md").string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
