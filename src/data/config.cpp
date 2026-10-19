#include "data/config.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "data/errors.hpp"

namespace sir {

AppConfig loadConfig(const std::filesystem::path& configFile) {
    std::ifstream in(configFile);
    if (!in) throw ConfigError("cannot open config file: " + configFile.string());
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        throw ConfigError("invalid JSON in " + configFile.string() + ": " + e.what());
    }
    try {
        AppConfig c;
        c.configFile = std::filesystem::absolute(configFile);
        c.root = c.configFile.parent_path().parent_path();
        c.modelName = j.at("model").at("name").get<std::string>();
        c.dim = j.at("model").at("embedding_dim").get<std::size_t>();
        if (c.dim == 0) throw ConfigError("model.embedding_dim must be > 0");
        c.normTolerance = j.at("embedding").at("norm_tolerance").get<double>();
        const auto& h = j.at("hnsw");
        c.hnsw.M = h.at("M").get<int>();
        c.hnsw.efConstruction = h.at("ef_construction").get<int>();
        c.hnsw.efSearch = h.at("ef_search").get<int>();
        c.hnsw.seed = h.at("seed").get<std::uint64_t>();
        c.hnsw.selectHeuristic = h.at("select_heuristic").get<bool>();
        c.hnsw.keepPrunedConnections = h.at("keep_pruned_connections").get<bool>();
        c.hnsw.protectLastInEdge = h.at("protect_last_in_edge").get<bool>();
        const auto& e = j.at("evaluation");
        c.evaluation.seed = e.at("seed").get<std::uint64_t>();
        c.evaluation.warmupQueries = e.at("warmup_queries").get<int>();
        c.evaluation.repetitions = e.at("repetitions").get<int>();
        c.evaluation.heldoutCaptionIndex = e.at("heldout_caption_index").get<int>();
        const auto& sweep = e.at("ef_search_sweep");
        if (!sweep.is_array())
            throw ConfigError("evaluation.ef_search_sweep in " + configFile.string() +
                              " must be an array");
        for (const auto& ef : sweep) c.evaluation.efSearchSweep.push_back(ef.get<int>());
        const auto& p = j.at("paths");
        auto path = [&](const char* key) { return c.root / p.at(key).get<std::string>(); };
        c.metadata = path("metadata");
        c.embeddings = path("embeddings");
        c.manifest = path("manifest");
        c.index = path("index");
        c.queries = path("queries");
        c.queryGroundTruth = path("query_ground_truth");
        c.queryEmbeddings = path("query_embeddings");
        c.queryManifest = path("query_manifest");
        c.benchmarkResults = path("benchmark_results");
        c.textEncoderServer = path("text_encoder_server");
        c.python = path("python");
        return c;
    } catch (const nlohmann::json::exception& ex) {
        throw ConfigError("missing or mistyped key in " + configFile.string() + ": " +
                          ex.what());
    }
}

}  // namespace sir
