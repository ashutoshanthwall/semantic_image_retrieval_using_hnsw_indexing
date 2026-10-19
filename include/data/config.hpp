#pragma once
// Typed view of config/config.json. Paths in the JSON are relative to the project root,
// which is the parent of the directory containing the config file.
#include <cstdint>
#include <filesystem>
#include <string>

#include "containers/vector.hpp"

namespace sir {

struct HnswParams {
    int M = 16;                   // max neighbours per node on layers >= 1 (2M on layer 0)
    int efConstruction = 200;     // candidate-list size during insertion
    int efSearch = 80;            // candidate-list size during queries
    std::uint64_t seed = 42;      // level-assignment RNG seed
    bool selectHeuristic = true;  // true: paper's SELECT_NEIGHBORS_HEURISTIC, false: naive top-M
    bool keepPrunedConnections = true;  // heuristic only: refill to M with discarded candidates
    bool protectLastInEdge = true;  // pruning never removes a node's only incoming edge
};

struct EvaluationParams {
    std::uint64_t seed = 0;
    int warmupQueries = 20;
    int repetitions = 5;
    int heldoutCaptionIndex = 0;  // caption used as the held-out query text for each image
    Vector<int> efSearchSweep;
};

struct AppConfig {
    std::filesystem::path root;
    std::string modelName;
    std::size_t dim = 0;
    double normTolerance = 1e-5;
    HnswParams hnsw;
    EvaluationParams evaluation;
    // Resolved absolute paths.
    std::filesystem::path metadata, embeddings, manifest, index, queries, queryGroundTruth,
        queryEmbeddings, queryManifest, benchmarkResults, textEncoderServer, python,
        configFile;
};

AppConfig loadConfig(const std::filesystem::path& configFile);

}  // namespace sir
