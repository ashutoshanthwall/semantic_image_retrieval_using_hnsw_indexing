#pragma once
#include <cstddef>
#include <filesystem>
#include <string>

namespace sir {

// Mirror of data/embeddings/manifest.json (and evaluation/query_manifest.json).
struct EmbeddingManifest {
    std::string model;
    std::size_t dim = 0;
    std::size_t rows = 0;
    bool l2Normalized = false;
    std::string dtype;
    std::string sha256;
    std::string createdUtc;
    std::string gitCommit;
};

EmbeddingManifest loadManifest(const std::filesystem::path& path);

}  // namespace sir
