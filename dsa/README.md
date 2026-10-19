# DSA files by member

Copies of the most DSA-heavy file each member owns, for the viva. These are snapshots and
are not compiled. The originals are the ones the build uses, so make any edits there.

| Member | File | Original | Core DSA |
|---|---|---|---|
| 1 | `member1/hash_map.hpp` | `include/containers/hash_map.hpp` | Hash table: open addressing, linear probing, FNV-1a, power-of-two capacity, rehash at load factor 0.7 |
| 1 | `member1/inverted_index.cpp` | `src/data/inverted_index.cpp` | Inverted index: sorted posting lists, two-pointer and galloping intersection (AND), k-way heap merge (OR) |
| 2 | `member2/sort.hpp` | `include/algorithms/sort.hpp` | Introsort: median-of-three quicksort, heap-sort fallback, insertion-sort finish |
| 3 | `member3/hnsw_index.cpp` | `src/index/hnsw_index.cpp` | HNSW graph: layered skip-list levels, two-heap best-first search, heuristic neighbour selection, pruning |
| 4 | `member4/vector.hpp` | `include/containers/vector.hpp` | Dynamic array: doubling growth (amortised O(1) push), manual memory management, strong exception guarantee |

Evidence: G12 (`evaluation/results/g12_caption_index.txt`) for Member 1, G6/G7 for Member 2,
G8/G9 for Member 3, `tests/test_containers.cpp` for Member 4.
