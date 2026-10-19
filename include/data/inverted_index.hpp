#pragma once
// InvertedIndex: for every caption word, the sorted list of image ids whose captions
// contain it (its posting list). The vocabulary is a HashMap from word to posting list.
//
//   build         one pass over the captions, O(total tokens). Documents arrive in id
//                 order, so every posting list is built already sorted and without
//                 duplicates: an id is appended only if it differs from the list's last one.
//   matchAll      AND query. Intersects the posting lists shortest first, so the running
//                 result never grows and the long lists are touched last. Each step is a
//                 two-pointer merge, O(a + b), or a galloping search, O(a log(b / a)), when
//                 one list is much shorter than the other.
//   matchAny      OR query. A k-way merge of the posting lists with a min-heap,
//                 O(n log k) for n postings in total.
//
// Words are runs of ASCII letters and digits (lowercased) or non-ASCII bytes, so UTF-8
// words stay whole; everything else separates words. A short list of stop words ("a",
// "the", ...) is dropped: they appear in nearly every caption, so their posting lists
// would hold nearly every id and filter nothing.
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include "containers/hash_map.hpp"
#include "containers/vector.hpp"
#include "data/metadata_manager.hpp"

namespace sir {

// Lowercased words of `text`, in order, stop words removed, duplicates kept.
Vector<std::string> tokenize(std::string_view text);

// Intersection of two sorted, duplicate-free id lists.
Vector<int> intersectMerge(std::span<const int> a, std::span<const int> b);
Vector<int> intersectGallop(std::span<const int> small, std::span<const int> large);

class InvertedIndex {
public:
    // Indexes every caption of every record except caption number `skipCaption` (-1 keeps
    // all). Passing evaluation.heldout_caption_index keeps the held-out query captions out
    // of the index, so a query can never match its own source caption.
    static InvertedIndex build(const MetadataManager& meta, int skipCaption);

    // Adds the words of `text` to document `id`. Ids must be added in non-decreasing order
    // (throws DataError otherwise); that is what keeps the posting lists sorted.
    void add(int id, std::string_view text);

    // Ids containing `term` (already lowercased), ascending; empty if the word is unknown.
    std::span<const int> postings(const std::string& term) const;
    // Ids whose captions contain every word of `query`. Empty if no word survives
    // tokenizing (e.g. the query is only stop words).
    Vector<int> matchAll(std::string_view query) const;
    // Ids whose captions contain at least one word of `query`, ascending.
    Vector<int> matchAny(std::string_view query) const;

    std::size_t vocabularySize() const noexcept { return terms_.size(); }
    std::size_t postingCount() const noexcept { return postingCount_; }
    const HashMap<std::string, Vector<int>>& terms() const noexcept { return terms_; }

private:
    Vector<std::span<const int>> queryLists(std::string_view query, bool& missing) const;

    HashMap<std::string, Vector<int>> terms_;
    std::size_t postingCount_ = 0;
    int lastId_ = -1;
};

}  // namespace sir
