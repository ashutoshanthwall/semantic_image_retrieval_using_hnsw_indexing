#include "index/retrieval_index.hpp"

#include <string>

#include "algorithms/sort.hpp"
#include "data/errors.hpp"
#include "similarity/vector_math.hpp"

namespace sir {

Vector<Candidate> toCandidates(const Vector<Neighbor>& closestFirst) {
    Vector<Candidate> out;


    out.reserve(closestFirst.size());
    for (const Neighbor& n : closestFirst) out.push_back({n.id, distanceToScore(n.dist)});
    // A total order (ids are unique), so the result does not depend on sort stability.
    sortRange(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {


        return a.score > b.score || (a.score == b.score && a.id < b.id);


    });


    return out;
}

void checkSearchArgs(std::span<const float> query, int k, std::size_t dim, std::size_t size) {
    if (query.size() != dim)


        throw QueryError("query dimension mismatch: got " + std::to_string(query.size()) +
                         ", index expects " + std::to_string(dim));


    if (!allFinite(query)) throw QueryError("query vector contains NaN or Inf");

    if (k <= 0) throw QueryError("K must be positive, got " + std::to_string(k));


    if (static_cast<std::size_t>(k) > size)


        throw QueryError("K=" + std::to_string(k) + " exceeds the number of indexed vectors (" +
                         std::to_string(size) + ")");
}

} 
