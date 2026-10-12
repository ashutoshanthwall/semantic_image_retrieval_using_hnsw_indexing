#pragma once
// Index lifecycle: build / save / load / validate / rebuild.
//
// File format (little-endian), version 1:
//   magic        8 bytes  "SIRHNSW\0"
//   version      u32
//   model        u32 length + UTF-8 bytes
//   dim          u64
//   store rows   u64      rows in the EmbeddingStore the index was built on
//   vectors      u64      nodes in the index
//   M, efConstruction, efSearch   i32 x3
//   seed         u64
//   heuristic, keepPrunedConnections, protectLastInEdge   u8 x3
//   source hash  64 bytes SHA-256 (hex) of the embedding file
//   entry, maxLevel   i32 x2
//   rng state    u32 length + text
//   payload size u64
//   payload      per node id 0..rows-1: i32 level, then per layer: u32 n, i32 ids[n]
//   checksum     32 bytes SHA-256 of every preceding byte
// Loading rejects: wrong magic/version, truncation, checksum mismatch (corruption), and a
// header that does not match the current embeddings (model, dim, row count, source hash).
#include <filesystem>
#include <string>

#include "containers/vector.hpp"
#include "data/config.hpp"
#include "embedding/embedding_store.hpp"
#include "index/hnsw_index.hpp"

namespace sir {

struct IndexValidation {
    std::size_t nodes = 0;
    std::size_t reachableLayer0 = 0;
    int maxLevel = -1;
    double meanDegreeLayer0 = 0.0;
    Vector<std::size_t> nodesPerLevel;
    Vector<int> unreachableLayer0;  // present nodes with no layer-0 path from the entry
    std::size_t inDegreeZeroLayer0 = 0;  // nodes that no other node links to on layer 0
};

struct BuildStats {
    double seconds = 0.0;
};

class IndexManager {
public:
    static HNSWIndex build(const EmbeddingStore& store, const HnswParams& params,
                           BuildStats* stats = nullptr);
    static void save(const HNSWIndex& index, const std::string& modelName,
                     const std::filesystem::path& path);
    static HNSWIndex load(const std::filesystem::path& path, const EmbeddingStore& store,
                          const std::string& modelName);
    // Structural checks (throws IndexError on a violated invariant) plus statistics.
    static IndexValidation validate(const HNSWIndex& index);
    static HNSWIndex rebuild(const EmbeddingStore& store, const HnswParams& params,
                             const std::string& modelName, const std::filesystem::path& path,
                             BuildStats* stats = nullptr);

    static constexpr std::uint32_t kFormatVersion = 1;
};

}  // namespace sir
