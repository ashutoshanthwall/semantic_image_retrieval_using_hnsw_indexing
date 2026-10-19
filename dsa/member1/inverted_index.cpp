#include "data/inverted_index.hpp"

#include "algorithms/sort.hpp"
#include "containers/binary_heap.hpp"
#include "data/errors.hpp"

namespace sir {

namespace {

constexpr std::string_view kStopWords[] = {"a",  "an", "and", "are", "at", "for", "in",
                                           "is", "of", "on",  "the", "to", "with"};

bool isStopWord(std::string_view w) {
    for (std::string_view s : kStopWords)
        if (w == s) return true;
    return false;
}

bool isWordByte(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
           c >= 0x80;
}

// First position in a[lo, n) whose value is >= x, found by doubling the step from lo
// (1, 2, 4, ...) until it passes x, then binary searching inside the last step.
std::size_t gallop(std::span<const int> a, std::size_t lo, int x) {
    std::size_t step = 1, hi = lo;
    while (hi < a.size() && a[hi] < x) {
        lo = hi + 1;
        hi += step;
        step *= 2;
    }
    if (hi > a.size()) hi = a.size();
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (a[mid] < x) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// One posting list's read position during the k-way merge.
struct Cursor {
    int id;
    std::size_t list, pos;
};
// BinaryHeap keeps the highest-ranked item on top; ranking larger ids lower makes it a
// min-heap on id.
struct LargerId {
    bool operator()(const Cursor& a, const Cursor& b) const { return a.id > b.id; }
};

}  // namespace

Vector<std::string> tokenize(std::string_view text) {
    Vector<std::string> words;
    std::string w;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        // One step past the end acts as a separator, which flushes the last word.
        const unsigned char c =
            i < text.size() ? static_cast<unsigned char>(text[i]) : static_cast<unsigned char>(' ');
        if (isWordByte(c)) {
            w += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : static_cast<char>(c);
        } else if (!w.empty()) {
            if (!isStopWord(w)) words.push_back(std::move(w));
            w.clear();
        }
    }
    return words;
}

Vector<int> intersectMerge(std::span<const int> a, std::span<const int> b) {
    Vector<int> out;
    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) ++i;
        else if (b[j] < a[i]) ++j;
        else {
            out.push_back(a[i]);
            ++i;
            ++j;
        }
    }
    return out;
}

Vector<int> intersectGallop(std::span<const int> small, std::span<const int> large) {
    Vector<int> out;
    std::size_t pos = 0;
    for (const int x : small) {
        pos = gallop(large, pos, x);  // ids only increase, so the search resumes at pos
        if (pos == large.size()) break;
        if (large[pos] == x) out.push_back(x);
    }
    return out;
}

InvertedIndex InvertedIndex::build(const MetadataManager& meta, int skipCaption) {
    InvertedIndex idx;
    for (std::size_t i = 0; i < meta.size(); ++i) {
        const ImageRecord& r = meta.at(static_cast<int>(i));
        for (std::size_t c = 0; c < r.captions.size(); ++c)
            if (static_cast<int>(c) != skipCaption) idx.add(static_cast<int>(i), r.captions[c]);
    }
    return idx;
}

void InvertedIndex::add(int id, std::string_view text) {
    if (id < lastId_)
        throw DataError("inverted index: id " + std::to_string(id) + " added after id " +
                        std::to_string(lastId_) + " (ids must be added in order)");
    lastId_ = id;
    for (const std::string& w : tokenize(text)) {
        Vector<int>& list = terms_[w];
        if (list.empty() || list.back() != id) {
            list.push_back(id);
            ++postingCount_;
        }
    }
}

std::span<const int> InvertedIndex::postings(const std::string& term) const {
    const Vector<int>* list = terms_.find(term);
    if (!list) return {};
    return {list->data(), list->size()};
}

// The posting list of each distinct query word. `missing` is set if some word is not in
// the vocabulary, which empties an AND query.
Vector<std::span<const int>> InvertedIndex::queryLists(std::string_view query,
                                                       bool& missing) const {
    Vector<std::string> words = tokenize(query);
    sortRange(words.begin(), words.end());
    Vector<std::span<const int>> lists;
    missing = false;
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (i > 0 && words[i] == words[i - 1]) continue;  // "dog and dog" is just "dog"
        const auto p = postings(words[i]);
        if (p.empty()) missing = true;
        else lists.push_back(p);
    }
    return lists;
}

Vector<int> InvertedIndex::matchAll(std::string_view query) const {
    bool missing = false;
    Vector<std::span<const int>> lists = queryLists(query, missing);
    if (missing || lists.empty()) return {};
    sortRange(lists.begin(), lists.end(), [](std::span<const int> a, std::span<const int> b) {
        return a.size() < b.size();
    });
    Vector<int> result(lists[0].data(), lists[0].data() + lists[0].size());
    for (std::size_t i = 1; i < lists.size() && !result.empty(); ++i) {
        const std::span<const int> cur(result.data(), result.size());
        // Galloping wins once the other list is much longer: it skips over runs of ids
        // instead of stepping through each one.
        result = lists[i].size() > 8 * cur.size() ? intersectGallop(cur, lists[i])
                                                  : intersectMerge(cur, lists[i]);
    }
    return result;
}

Vector<int> InvertedIndex::matchAny(std::string_view query) const {
    bool missing = false;
    const Vector<std::span<const int>> lists = queryLists(query, missing);
    BinaryHeap<Cursor, LargerId> heap;
    heap.reserve(lists.size());
    for (std::size_t l = 0; l < lists.size(); ++l) heap.push({lists[l][0], l, 0});
    Vector<int> out;
    while (!heap.empty()) {
        Cursor c = heap.top();
        heap.pop();
        if (out.empty() || out.back() != c.id) out.push_back(c.id);  // id in several lists
        if (++c.pos < lists[c.list].size()) {
            c.id = lists[c.list][c.pos];
            heap.push(c);
        }
    }
    return out;
}

}  // namespace sir
