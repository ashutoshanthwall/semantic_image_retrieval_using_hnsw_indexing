#pragma once
// Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016/2018), written from
// scratch. Nodes store only ids and adjacency lists; vectors are read from the
// EmbeddingStore through spans, never copied.
//
//   level(new node) = floor(-ln(U) * mL),  U ~ (0,1] from a seeded mt19937_64, mL = 1/ln(M)
//   insert:  greedy descent on layers > level(new); then on each layer <= level(new):
//            SEARCH-LAYER with efConstruction -> SELECT-NEIGHBORS (M) -> bidirectional
//            links -> prune any neighbour list that exceeds Mmax (M, or 2M on layer 0)
//   search:  greedy descent to layer 1, SEARCH-LAYER on layer 0 with ef = max(efSearch, k)
//
// Every queue, comparison and pruning decision uses the (dist, id) order from topk.hpp,
// dist = 1 - dot, lower is better.
//
// Not thread-safe: search() reuses a mutable visited-marker array. The benchmark protocol
// is single-threaded by design.
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "data/config.hpp"
#include "embedding/embedding_store.hpp"
#include "index/retrieval_index.hpp"

namespace sir {

class HNSWIndex final : public RetrievalIndex {
public:
    HNSWIndex(const EmbeddingStore& store, const HnswParams& params);
    HNSWIndex(HNSWIndex&&) noexcept = default;
    HNSWIndex& operator=(HNSWIndex&&) noexcept = default;
    HNSWIndex(const HNSWIndex&) = delete;
    HNSWIndex& operator=(const HNSWIndex&) = delete;

    void insert(int id, std::span<const float> embedding) override;
    std::vector<Candidate> search(std::span<const float> query, int k) const override;
    std::size_t size() const override { return count_; }
    std::size_t memoryBytes() const override;

    void setEfSearch(int ef);
    const HnswParams& params() const noexcept { return params_; }
    const EmbeddingStore& store() const noexcept { return *store_; }

    // Introspection (validation, persistence, tests).
    int entryPoint() const noexcept { return entry_; }
    int maxLevel() const noexcept { return maxLevel_; }
    bool contains(int id) const noexcept;
    int levelOf(int id) const;  // -1 if absent
    std::span<const int> neighbors(int id, int layer) const;
    // Incrementally maintained in-degree (used by protectLastInEdge); tests recount it.
    std::uint32_t inDegreeOf(int id, int layer) const {
        return inDeg_.at(static_cast<std::size_t>(id)).at(static_cast<std::size_t>(layer));
    }
    int maxDegree(int layer) const noexcept { return layer == 0 ? 2 * params_.M : params_.M; }
    // Number of distance evaluations since the last reset (analysis only).
    std::uint64_t distanceComputations() const noexcept { return distCount_; }
    void resetDistanceComputations() const noexcept { distCount_ = 0; }

    // Persistence support: RNG state as text, and raw graph restoration.
    std::string rngState() const;
    struct NodeData {
        int level = -1;
        std::vector<std::vector<int>> links;  // links[layer]
    };
    static HNSWIndex restore(const EmbeddingStore& store, const HnswParams& params,
                             std::vector<NodeData> nodes, int entry, int maxLevel,
                             const std::string& rngState);

private:
    float dist(std::span<const float> q, int id) const noexcept;
    int randomLevel();
    Neighbor greedyClosest(std::span<const float> q, Neighbor ep, int layer) const;
    // SEARCH-LAYER (paper Alg. 2). Returns up to ef results, closest first.
    std::vector<Neighbor> searchLayer(std::span<const float> q,
                                      const std::vector<Neighbor>& entryPoints, int ef,
                                      int layer) const;
    // SELECT-NEIGHBORS: naive (M closest) or heuristic (paper Alg. 4, extendCandidates off,
    // keepPrunedConnections configurable). `candidates` must be closest first,
    // with distances measured from the base element.
    std::vector<Neighbor> selectNeighbors(const std::vector<Neighbor>& candidates, int m) const;
    // Prunes id's list on `layer` back to Mmax. With protectLastInEdge, an edge that is its
    // target's only incoming edge on this layer is kept in place of the farthest kept edge
    // whose target has another incoming edge (an addition of this project, not in the paper).
    void shrinkNeighbors(int id, int layer);
    std::uint32_t& inDegree(int id, int layer) {
        return inDeg_[static_cast<std::size_t>(id)][static_cast<std::size_t>(layer)];
    }
    void nextVisitEpoch() const;

    const EmbeddingStore* store_;  // non-owning
    HnswParams params_;
    double mL_;
    std::mt19937_64 rng_;
    std::vector<NodeData> nodes_;  // indexed by id; level -1 = not inserted
    std::vector<bool> present_;
    std::vector<std::vector<std::uint32_t>> inDeg_;  // inDeg_[id][layer]
    std::size_t count_ = 0;
    int entry_ = -1;
    int maxLevel_ = -1;

    mutable std::vector<std::uint32_t> visited_;  // visited_[id] == epoch -> visited
    mutable std::uint32_t epoch_ = 0;
    mutable std::uint64_t distCount_ = 0;
};

}  // namespace sir
