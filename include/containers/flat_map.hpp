#pragma once
// FlatMap<K, V>: an associative array kept as a Vector of entries sorted by key. Lookup is
// a binary search, O(log n); inserting a new key is O(n). Meant for small maps such as
// command-line options. K needs operator<.
#include <cstddef>

#include "containers/vector.hpp"

namespace sir {

template <typename K, typename V>
class FlatMap {
public:
    struct Entry {
        K key;
        V value;
    };

    // The value for `key`, inserting a value-initialised one first if the key is new.
    V& operator[](const K& key) {
        const std::size_t pos = lowerBound(key);
        if (pos == entries_.size() || key < entries_[pos].key)
            entries_.insertAt(pos, Entry{key, V{}});
        return entries_[pos].value;
    }
    // Pointer to the value for `key`, or nullptr if absent.
    const V* find(const K& key) const {
        const std::size_t pos = lowerBound(key);
        if (pos == entries_.size() || key < entries_[pos].key) return nullptr;
        return &entries_[pos].value;
    }
    bool contains(const K& key) const { return find(key) != nullptr; }

    std::size_t size() const noexcept { return entries_.size(); }
    const Entry* begin() const noexcept { return entries_.begin(); }
    const Entry* end() const noexcept { return entries_.end(); }

private:
    std::size_t lowerBound(const K& key) const {
        std::size_t lo = 0, hi = entries_.size();
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (entries_[mid].key < key) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    Vector<Entry> entries_;
};

}  // namespace sir
