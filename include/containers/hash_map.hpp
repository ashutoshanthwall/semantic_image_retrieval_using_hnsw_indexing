#pragma once
// HashMap<K, V, Hash>: an unordered map using open addressing with linear probing. All
// entries live in one slot array whose length is a power of two, so a key's home slot is
// hash(key) & (capacity - 1); on a collision the key goes in the next free slot after it
// (wrapping around), and lookups walk the same path until they hit the key or a free slot.
//
//   find / operator[]   expected O(1) while the load factor (size / capacity) stays at
//                       most 0.7; the worst case, every key in one cluster, is O(n)
//   growth              an insert that would push the load factor above 0.7 first doubles
//                       the capacity and re-inserts every key (a rehash, O(n)), so n
//                       inserts cost O(n) in total, amortised O(1) each
//
// There is no erase: the inverted index only ever adds keys. Deleting from a linear-probing
// table needs tombstones or backward-shift deletion, because simply emptying a slot would
// cut the probe path of every key stored after it.
// K needs operator== and must be default-constructible, as must V.
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "containers/vector.hpp"

namespace sir {

template <typename K>
struct Hash;

// FNV-1a, 64-bit: xor in each byte, then multiply by the FNV prime.
template <>
struct Hash<std::string> {
    std::uint64_t operator()(const std::string& s) const noexcept {
        std::uint64_t h = 0xcbf29ce484222325ull;
        for (const char c : s) {
            h ^= static_cast<unsigned char>(c);
            h *= 0x100000001b3ull;
        }
        return h;
    }
};

// SplitMix64 finaliser: spreads consecutive integers over all 64 bits, so masking off
// the low bits still separates them.
template <>
struct Hash<int> {
    std::uint64_t operator()(int key) const noexcept {
        std::uint64_t x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key));
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }
};

template <typename K, typename V, typename H = Hash<K>>
class HashMap {
public:
    // The value for `key`, inserting a value-initialised one first if the key is new.
    V& operator[](const K& key) {
        if (10 * (size_ + 1) > 7 * slots_.size()) grow();
        std::size_t i = homeSlot(key);
        for (; slots_[i].used; i = (i + 1) & mask())
            if (slots_[i].key == key) return slots_[i].value;
        slots_[i].used = true;
        slots_[i].key = key;
        ++size_;
        return slots_[i].value;
    }
    // Pointer to the value for `key`, or nullptr if absent.
    const V* find(const K& key) const {
        if (slots_.empty()) return nullptr;
        for (std::size_t i = homeSlot(key); slots_[i].used; i = (i + 1) & mask())
            if (slots_[i].key == key) return &slots_[i].value;
        return nullptr;
    }
    bool contains(const K& key) const { return find(key) != nullptr; }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    std::size_t capacity() const noexcept { return slots_.size(); }
    double loadFactor() const noexcept {
        return slots_.empty() ? 0.0 : static_cast<double>(size_) / static_cast<double>(slots_.size());
    }

    // Mean number of slots a successful lookup inspects (1 = every key in its home slot).
    // Linear-probing theory predicts about (1 + 1/(1 - a)) / 2 at load factor a.
    double averageProbeLength() const noexcept {
        if (size_ == 0) return 0.0;
        std::size_t total = 0;
        for (std::size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].used) total += ((i - homeSlot(slots_[i].key)) & mask()) + 1;
        return static_cast<double>(total) / static_cast<double>(size_);
    }

    // Calls f(key, value) for every entry, in slot order (which is not key order).
    template <typename F>
    void forEach(F f) const {
        for (const Slot& s : slots_)
            if (s.used) f(s.key, s.value);
    }

private:
    struct Slot {
        K key{};
        V value{};
        bool used = false;
    };

    std::size_t mask() const noexcept { return slots_.size() - 1; }
    std::size_t homeSlot(const K& key) const noexcept {
        return static_cast<std::size_t>(hash_(key)) & mask();
    }

    void grow() {
        Vector<Slot> old = std::move(slots_);
        slots_ = Vector<Slot>(old.empty() ? 8 : 2 * old.size());
        for (Slot& s : old) {
            if (!s.used) continue;
            std::size_t i = homeSlot(s.key);
            while (slots_[i].used) i = (i + 1) & mask();
            slots_[i] = std::move(s);
        }
    }

    Vector<Slot> slots_;
    std::size_t size_ = 0;
    H hash_{};
};

}  // namespace sir
