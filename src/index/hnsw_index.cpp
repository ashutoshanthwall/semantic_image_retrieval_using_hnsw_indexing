#include "index/hnsw_index.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "data/errors.hpp"
#include "index/brute_force_index.hpp"  // checkInsert
#include "similarity/vector_math.hpp"

namespace sir {

HNSWIndex::HNSWIndex(const EmbeddingStore& store, const HnswParams& params)
    : store_(&store),
      params_(params),
      mL_(0.0),
      rng_(params.seed),
      nodes_(store.size()),
      present_(store.size(), false),
      inDeg_(store.size()),
      visited_(store.size(), 0) {
    if (params_.M < 2) throw IndexError("HNSW M must be >= 2, got " + std::to_string(params_.M));
    if (params_.efConstruction < 1)
        throw IndexError("HNSW efConstruction must be >= 1");
    if (params_.efSearch < 1) throw IndexError("HNSW efSearch must be >= 1");
    mL_ = 1.0 / std::log(static_cast<double>(params_.M));
}

void HNSWIndex::setEfSearch(int ef) {
    if (ef < 1) throw IndexError("efSearch must be >= 1, got " + std::to_string(ef));
    params_.efSearch = ef;
}

float HNSWIndex::dist(std::span<const float> q, int id) const noexcept {
    ++distCount_;
    return distance(q, store_->row(static_cast<std::size_t>(id)));
}

int HNSWIndex::randomLevel() {
    // U in (0, 1] built from the top 53 bits: portable across standard libraries
    // (std::uniform_real_distribution is implementation-defined).
    const double u = (static_cast<double>(rng_() >> 11) + 1.0) * 0x1.0p-53;
    return static_cast<int>(std::floor(-std::log(u) * mL_));
}

void HNSWIndex::nextVisitEpoch() const {
    if (++epoch_ == 0) {  // wrapped: clear markers once every 2^32 searches
        std::fill(visited_.begin(), visited_.end(), 0u);
        epoch_ = 1;
    }
}

Neighbor HNSWIndex::greedyClosest(std::span<const float> q, Neighbor ep, int layer) const {
    // Upper-layer routing (equivalent to SEARCH-LAYER with ef = 1): move to the closest
    // neighbour while that strictly improves on the current node.
    for (bool improved = true; improved;) {
        improved = false;
        Neighbor best = ep;
        for (int e : nodes_[static_cast<std::size_t>(ep.id)].links[static_cast<std::size_t>(layer)]) {
            const Neighbor n{dist(q, e), e};
            if (closer(n, best)) best = n;
        }
        if (best.id != ep.id) {
            ep = best;
            improved = true;
        }
    }
    return ep;
}

std::vector<Neighbor> HNSWIndex::searchLayer(std::span<const float> q,
                                             const std::vector<Neighbor>& entryPoints, int ef,
                                             int layer) const {
    nextVisitEpoch();
    MinDistHeap candidates;  // C: closest on top — next node to expand
    MaxDistHeap results;     // W: farthest on top — the current worst of the best ef
    for (const Neighbor& e : entryPoints) {
        visited_[static_cast<std::size_t>(e.id)] = epoch_;
        candidates.push(e);
        results.push(e);
        if (results.size() > static_cast<std::size_t>(ef)) results.pop();
    }
    while (!candidates.empty()) {
        const Neighbor c = candidates.top();
        // Every remaining candidate is farther than the worst kept result: stop.
        if (closer(results.top(), c)) break;
        candidates.pop();
        for (int e : nodes_[static_cast<std::size_t>(c.id)].links[static_cast<std::size_t>(layer)]) {
            auto& mark = visited_[static_cast<std::size_t>(e)];
            if (mark == epoch_) continue;
            mark = epoch_;
            const Neighbor n{dist(q, e), e};
            if (results.size() < static_cast<std::size_t>(ef) || closer(n, results.top())) {
                candidates.push(n);
                results.push(n);
                if (results.size() > static_cast<std::size_t>(ef)) results.pop();
            }
        }
    }
    std::vector<Neighbor> out(results.size());
    for (std::size_t i = out.size(); i-- > 0;) {  // max-heap pops farthest first
        out[i] = results.top();
        results.pop();
    }
    return out;
}

std::vector<Neighbor> HNSWIndex::selectNeighbors(const std::vector<Neighbor>& candidates,
                                                 int m) const {
    const auto limit = static_cast<std::size_t>(m);
    if (!params_.selectHeuristic)  // naive: the m closest
        return {candidates.begin(),
                candidates.begin() + static_cast<long>(std::min(limit, candidates.size()))};
    // Heuristic: walk candidates closest-first; keep e only if it is closer to the base
    // element than to every neighbour already kept. This favours neighbours in
    // different directions, which preserves long-range connectivity between clusters.
    // As in the paper (and unlike hnswlib's early return), it is applied even when there
    // are fewer than m candidates.
    std::vector<Neighbor> kept, discarded;
    kept.reserve(limit);
    for (const Neighbor& e : candidates) {
        if (kept.size() >= limit) break;
        const auto ev = store_->row(static_cast<std::size_t>(e.id));
        bool good = true;
        for (const Neighbor& r : kept) {
            ++distCount_;
            if (distance(ev, store_->row(static_cast<std::size_t>(r.id))) < e.dist) {
                good = false;
                break;
            }
        }
        if (good) kept.push_back(e);
        else if (params_.keepPrunedConnections) discarded.push_back(e);
    }
    // keepPrunedConnections (paper Alg. 4): top up to m with the closest discarded
    // candidates. Without it, an outlier whose candidates all lie in one tight cluster gets
    // a single out-link, hence a single in-link, which later pruning can remove, leaving the
    // node unreachable. `discarded` is already closest-first.
    for (std::size_t i = 0; i < discarded.size() && kept.size() < limit; ++i)
        kept.push_back(discarded[i]);
    return kept;
}

void HNSWIndex::shrinkNeighbors(int id, int layer) {
    auto& links = nodes_[static_cast<std::size_t>(id)].links[static_cast<std::size_t>(layer)];
    const auto base = store_->row(static_cast<std::size_t>(id));
    std::vector<Neighbor> cands;
    cands.reserve(links.size());
    for (int e : links) cands.push_back({dist(base, e), e});
    std::sort(cands.begin(), cands.end(), closer);
    std::vector<Neighbor> kept = selectNeighbors(cands, maxDegree(layer));
    auto isKept = [&kept](int x) {
        return std::any_of(kept.begin(), kept.end(), [x](const Neighbor& n) { return n.id == x; });
    };
    if (params_.protectLastInEdge) {
        for (const Neighbor& c : cands) {  // closest first: deterministic
            if (isKept(c.id) || inDegree(c.id, layer) > 1) continue;
            // Dropping id -> c.id would leave c.id with no incoming edge on this layer.
            if (kept.size() < static_cast<std::size_t>(maxDegree(layer))) {
                kept.push_back(c);
                continue;
            }
            for (auto it = kept.rbegin(); it != kept.rend(); ++it)
                if (inDegree(it->id, layer) > 1) {  // its target keeps another in-edge
                    *it = c;
                    break;
                }
        }
        std::sort(kept.begin(), kept.end(), closer);
    }
    for (int e : links)
        if (!isKept(e)) --inDegree(e, layer);
    links.clear();
    for (const Neighbor& n : kept) links.push_back(n.id);
}

void HNSWIndex::insert(int id, std::span<const float> embedding) {
    checkInsert(*store_, id, embedding, present_);
    const auto q = store_->row(static_cast<std::size_t>(id));  // same values, store-owned
    const int level = randomLevel();
    NodeData& node = nodes_[static_cast<std::size_t>(id)];
    node.level = level;
    node.links.resize(static_cast<std::size_t>(level) + 1);
    for (int l = 0; l <= level; ++l)
        node.links[static_cast<std::size_t>(l)].reserve(static_cast<std::size_t>(maxDegree(l)) + 1);
    inDeg_[static_cast<std::size_t>(id)].assign(static_cast<std::size_t>(level) + 1, 0);
    present_[static_cast<std::size_t>(id)] = true;
    ++count_;

    if (entry_ < 0) {
        entry_ = id;
        maxLevel_ = level;
        return;
    }

    Neighbor ep{dist(q, entry_), entry_};
    for (int lc = maxLevel_; lc > level; --lc) ep = greedyClosest(q, ep, lc);

    std::vector<Neighbor> entryPoints{ep};
    for (int lc = std::min(level, maxLevel_); lc >= 0; --lc) {
        std::vector<Neighbor> w = searchLayer(q, entryPoints, params_.efConstruction, lc);
        const std::vector<Neighbor> chosen = selectNeighbors(w, params_.M);
        auto& mine = node.links[static_cast<std::size_t>(lc)];
        for (const Neighbor& n : chosen) {
            mine.push_back(n.id);
            ++inDegree(n.id, lc);
        }
        for (const Neighbor& n : chosen) {  // bidirectional links, then prune to Mmax
            auto& theirs = nodes_[static_cast<std::size_t>(n.id)].links[static_cast<std::size_t>(lc)];
            theirs.push_back(id);
            ++inDegree(id, lc);
            if (theirs.size() > static_cast<std::size_t>(maxDegree(lc))) shrinkNeighbors(n.id, lc);
        }
        entryPoints = std::move(w);
    }
    if (level > maxLevel_) {
        maxLevel_ = level;
        entry_ = id;
    }
}

std::vector<Candidate> HNSWIndex::search(std::span<const float> query, int k) const {
    checkSearchArgs(query, k, store_->dim(), count_);
    Neighbor ep{dist(query, entry_), entry_};
    for (int lc = maxLevel_; lc > 0; --lc) ep = greedyClosest(query, ep, lc);
    std::vector<Neighbor> w = searchLayer(query, {ep}, std::max(params_.efSearch, k), 0);
    if (w.size() > static_cast<std::size_t>(k)) w.resize(static_cast<std::size_t>(k));
    return toCandidates(w);
}

bool HNSWIndex::contains(int id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < present_.size() &&
           present_[static_cast<std::size_t>(id)];
}

int HNSWIndex::levelOf(int id) const {
    return contains(id) ? nodes_[static_cast<std::size_t>(id)].level : -1;
}

std::span<const int> HNSWIndex::neighbors(int id, int layer) const {
    if (!contains(id) || layer < 0 || layer > nodes_[static_cast<std::size_t>(id)].level)
        throw IndexError("neighbors: node " + std::to_string(id) + " has no layer " +
                         std::to_string(layer));
    return nodes_[static_cast<std::size_t>(id)].links[static_cast<std::size_t>(layer)];
}

std::size_t HNSWIndex::memoryBytes() const {
    std::size_t bytes = nodes_.capacity() * sizeof(NodeData) + present_.capacity() / 8 +
                        visited_.capacity() * sizeof(std::uint32_t) +
                        inDeg_.capacity() * sizeof(std::vector<std::uint32_t>);
    for (const auto& d : inDeg_) bytes += d.capacity() * sizeof(std::uint32_t);
    for (const NodeData& n : nodes_) {
        bytes += n.links.capacity() * sizeof(std::vector<int>);
        for (const auto& l : n.links) bytes += l.capacity() * sizeof(int);
    }
    return bytes;
}

std::string HNSWIndex::rngState() const {
    std::ostringstream os;
    os << rng_;
    return os.str();
}

HNSWIndex HNSWIndex::restore(const EmbeddingStore& store, const HnswParams& params,
                             std::vector<NodeData> nodes, int entry, int maxLevel,
                             const std::string& rngState) {
    HNSWIndex idx(store, params);
    if (nodes.size() != store.size())
        throw IndexError("restore: node table size != store size");
    idx.nodes_ = std::move(nodes);
    for (std::size_t i = 0; i < idx.nodes_.size(); ++i) {
        if (idx.nodes_[i].level >= 0) {
            idx.present_[i] = true;
            ++idx.count_;
            idx.inDeg_[i].assign(static_cast<std::size_t>(idx.nodes_[i].level) + 1, 0);
        }
    }
    for (const NodeData& n : idx.nodes_)  // in-degrees are derived, not persisted
        for (std::size_t l = 0; l < n.links.size(); ++l)
            for (int e : n.links[l]) {
                if (e < 0 || static_cast<std::size_t>(e) >= idx.nodes_.size() ||
                    idx.nodes_[static_cast<std::size_t>(e)].level < static_cast<int>(l))
                    throw IndexError("restore: link to a node absent from layer " +
                                     std::to_string(l));
                ++idx.inDeg_[static_cast<std::size_t>(e)][l];
            }
    idx.entry_ = entry;
    idx.maxLevel_ = maxLevel;
    std::istringstream is(rngState);
    is >> idx.rng_;
    if (!is) throw IndexError("restore: invalid RNG state");
    return idx;
}

}  // namespace sir
