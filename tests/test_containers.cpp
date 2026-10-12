#include <gtest/gtest.h>

#include <random>
#include <span>
#include <string>

#include "algorithms/basic.hpp"
#include "algorithms/sort.hpp"
#include "containers/binary_heap.hpp"
#include "containers/bit_vector.hpp"
#include "containers/fixed_array.hpp"
#include "containers/flat_map.hpp"
#include "containers/flat_set.hpp"
#include "containers/vector.hpp"

using namespace sir;

namespace {

// Counts live instances, so a test can check that every constructed element is destroyed.
struct Tracked {
    static int live;
    int v;
    Tracked(int x) : v(x) { ++live; }  // NOLINT: implicit on purpose
    Tracked(const Tracked& o) : v(o.v) { ++live; }
    Tracked(Tracked&& o) noexcept : v(o.v) { ++live; }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&) noexcept = default;
    ~Tracked() { --live; }
};
int Tracked::live = 0;

// Its move may throw, so Vector must relocate by copying; the copy throws on demand.
struct ThrowOnCopy {
    static int copiesBeforeThrow;
    static int live;
    int v;
    explicit ThrowOnCopy(int x) : v(x) { ++live; }
    ThrowOnCopy(const ThrowOnCopy& o) : v(o.v) {
        if (copiesBeforeThrow-- == 0) throw std::runtime_error("copy failed");
        ++live;
    }
    ThrowOnCopy(ThrowOnCopy&& o) noexcept(false) : ThrowOnCopy(static_cast<const ThrowOnCopy&>(o)) {}
    ~ThrowOnCopy() { --live; }
};
int ThrowOnCopy::copiesBeforeThrow = 1 << 30;
int ThrowOnCopy::live = 0;

template <typename T, typename Less>
bool isSorted(const Vector<T>& v, Less less) {
    for (std::size_t i = 1; i < v.size(); ++i)
        if (less(v[i], v[i - 1])) return false;
    return true;
}

// Values are in [0, range): equal counts per value means `b` is a permutation of `a`.
bool samePermutation(const Vector<int>& a, const Vector<int>& b, int range) {
    if (a.size() != b.size()) return false;
    Vector<long> count(static_cast<std::size_t>(range), 0);
    for (int x : a) ++count[static_cast<std::size_t>(x)];
    for (int x : b) --count[static_cast<std::size_t>(x)];
    for (long c : count)
        if (c != 0) return false;
    return true;
}

}  // namespace

TEST(Vector, PushBackDoublesCapacity) {
    Vector<int> v;
    EXPECT_TRUE(v.empty());
    EXPECT_EQ(v.capacity(), 0u);
    Vector<std::size_t> seen;
    for (int i = 0; i < 100; ++i) {
        v.push_back(i);
        if (seen.empty() || seen.back() != v.capacity()) seen.push_back(v.capacity());
    }
    const Vector<std::size_t> expected{1, 2, 4, 8, 16, 32, 64, 128};
    EXPECT_EQ(seen, expected);
    for (int i = 0; i < 100; ++i) ASSERT_EQ(v[static_cast<std::size_t>(i)], i);
    EXPECT_EQ(v.front(), 0);
    EXPECT_EQ(v.back(), 99);
}

TEST(Vector, ReserveResizeAssignClear) {
    Vector<int> v;
    v.reserve(33);
    EXPECT_EQ(v.capacity(), 33u);  // exact: index memory accounting depends on it
    for (int i = 0; i < 33; ++i) v.push_back(i);
    EXPECT_EQ(v.capacity(), 33u);
    v.reserve(10);  // never shrinks
    EXPECT_EQ(v.capacity(), 33u);

    Vector<int> r(5);  // value-initialised
    for (int x : r) EXPECT_EQ(x, 0);
    r.resize(8, 7);
    EXPECT_EQ(r.size(), 8u);
    EXPECT_EQ(r[4], 0);
    EXPECT_EQ(r[7], 7);
    r.resize(2);
    EXPECT_EQ(r.size(), 2u);
    EXPECT_EQ(r.capacity(), 10u);  // grew from 5 to max(8, 2*5)

    r.assign(3, 9);
    EXPECT_EQ(r.size(), 3u);
    for (int x : r) EXPECT_EQ(x, 9);
    r.assign(r.size(), r[0]);  // value aliases an element
    EXPECT_EQ(r[2], 9);
    r.clear();
    EXPECT_TRUE(r.empty());
    EXPECT_EQ(r.capacity(), 10u);  // clear keeps the buffer

    EXPECT_THROW(r.at(0), std::out_of_range);
}

TEST(Vector, CopyMoveAndEquality) {
    Vector<std::string> a{"x", "yy", "zzz"};
    Vector<std::string> b = a;
    EXPECT_EQ(a, b);
    b[1] = "changed";
    EXPECT_EQ(a[1], "yy");  // deep copy
    EXPECT_FALSE(a == b);

    Vector<std::string> c = std::move(b);
    EXPECT_TRUE(b.empty());  // NOLINT(bugprone-use-after-move): moved-from is empty
    EXPECT_EQ(c[1], "changed");

    Vector<std::string> d;
    d = a;
    EXPECT_EQ(d, a);
    d = std::move(c);
    EXPECT_EQ(d[1], "changed");
    d = d;  // self-assignment is a no-op
    EXPECT_EQ(d.size(), 3u);
}

TEST(Vector, PushBackOfOwnElementSurvivesReallocation) {
    Vector<std::string> v{"first-element-long-enough-to-be-heap-allocated"};
    for (int i = 0; i < 10; ++i) v.push_back(v[0]);  // reallocates while reading v[0]
    for (const auto& s : v) EXPECT_EQ(s, v[0]);
    Vector<int> w{1, 2, 3};
    w.append(w.data(), w.size());  // append a copy of itself, forcing growth
    const Vector<int> expected{1, 2, 3, 1, 2, 3};
    EXPECT_EQ(w, expected);
}

TEST(Vector, InsertAt) {
    Vector<int> v{1, 3, 5};
    v.insertAt(0, 0);
    v.insertAt(2, 2);
    v.insertAt(v.size(), 6);
    const Vector<int> expected{0, 1, 2, 3, 5, 6};
    EXPECT_EQ(v, expected);
}

TEST(Vector, EveryElementIsDestroyed) {
    {
        Vector<Tracked> v;
        for (int i = 0; i < 1000; ++i) v.push_back(Tracked(i));
        v.resize(10, Tracked(0));  // shrinks: destroys 990 elements
        v.pop_back();
        Vector<Tracked> copy = v;
        copy.insertAt(3, Tracked(-1));
        v.clear();
        EXPECT_EQ(Tracked::live, static_cast<int>(copy.size()));
    }
    EXPECT_EQ(Tracked::live, 0);
}

TEST(Vector, ThrowingCopyDuringGrowthLeavesVectorUnchanged) {
    {
        Vector<ThrowOnCopy> v;
        v.reserve(4);
        for (int i = 0; i < 4; ++i) v.emplace_back(i);
        ThrowOnCopy::copiesBeforeThrow = 2;  // the third copy during relocation throws
        EXPECT_THROW(v.emplace_back(4), std::runtime_error);
        ThrowOnCopy::copiesBeforeThrow = 1 << 30;
        ASSERT_EQ(v.size(), 4u);
        EXPECT_EQ(v.capacity(), 4u);
        for (int i = 0; i < 4; ++i) EXPECT_EQ(v[static_cast<std::size_t>(i)].v, i);
        EXPECT_EQ(ThrowOnCopy::live, 4);  // no leaked partial copies
    }
    EXPECT_EQ(ThrowOnCopy::live, 0);
}

TEST(Vector, ConvertsToSpan) {
    Vector<float> v{1.0f, 2.0f, 3.0f};
    std::span<const float> s = v;
    EXPECT_EQ(s.size(), 3u);
    EXPECT_EQ(s.data(), v.data());
    std::span<float> m = v;
    m[0] = 9.0f;
    EXPECT_EQ(v[0], 9.0f);
}

TEST(BitVector, SetAndReadAcrossWordBoundaries) {
    BitVector b(130, false);
    EXPECT_EQ(b.size(), 130u);
    EXPECT_EQ(b.capacity(), 192u);  // three 64-bit words
    for (std::size_t i : {0u, 63u, 64u, 127u, 129u}) b.set(i, true);
    for (std::size_t i = 0; i < 130; ++i)
        EXPECT_EQ(b[i], i == 0 || i == 63 || i == 64 || i == 127 || i == 129) << i;
    b.set(64, false);
    EXPECT_FALSE(b[64]);
    EXPECT_TRUE(b[63]);
    BitVector ones(70, true);
    for (std::size_t i = 0; i < 70; ++i) EXPECT_TRUE(ones[i]);
}

TEST(BinaryHeap, PopsInPriorityOrder) {
    struct Less {
        bool operator()(int a, int b) const { return a < b; }
    };
    std::mt19937_64 rng(1);
    BinaryHeap<int, Less> h;
    Vector<int> in;
    for (int i = 0; i < 2000; ++i) {
        const int x = static_cast<int>(rng() % 500);  // many duplicates
        in.push_back(x);
        h.push(x);
    }
    Vector<int> out;
    while (!h.empty()) {
        out.push_back(h.top());
        h.pop();
    }
    EXPECT_TRUE(isSorted(out, [](int a, int b) { return a > b; }));  // max first
    EXPECT_TRUE(samePermutation(in, out, 500));
}

TEST(Sort, RandomAndAdversarialInputs) {
    std::mt19937_64 rng(7);
    auto check = [](Vector<int> v, int range, const char* what) {
        const Vector<int> before = v;
        sortRange(v.begin(), v.end());
        EXPECT_TRUE(isSorted(v, [](int a, int b) { return a < b; })) << what << " n=" << v.size();
        EXPECT_TRUE(samePermutation(before, v, range)) << what << " n=" << v.size();
    };
    for (std::size_t n = 0; n <= 300; ++n) {
        Vector<int> random, fewValues, ascending, descending, equal, organPipe;
        for (std::size_t i = 0; i < n; ++i) {
            random.push_back(static_cast<int>(rng() % 1000));
            fewValues.push_back(static_cast<int>(rng() % 3));
            ascending.push_back(static_cast<int>(i));
            descending.push_back(static_cast<int>(n - i));
            equal.push_back(5);
            organPipe.push_back(static_cast<int>(i < n / 2 ? i : n - i));
        }
        check(random, 1000, "random");
        check(fewValues, 3, "few values");
        check(ascending, 1000, "ascending");
        check(descending, 1000, "descending");
        check(equal, 1000, "all equal");
        check(organPipe, 1000, "organ pipe");
    }
    Vector<int> big;
    for (int i = 0; i < 100000; ++i) big.push_back(static_cast<int>(rng() % 100000));
    check(big, 100000, "random 1e5");
}

TEST(Sort, CustomComparatorAndHeapSortFallback) {
    struct Item {
        float key;
        int id;
    };
    auto byKeyThenId = [](const Item& a, const Item& b) {
        return a.key < b.key || (a.key == b.key && a.id < b.id);
    };
    std::mt19937_64 rng(3);
    Vector<Item> items;
    for (int i = 0; i < 5000; ++i) items.push_back({static_cast<float>(rng() % 50), i});
    Vector<Item> viaHeap = items;
    sortRange(items.begin(), items.end(), byKeyThenId);
    EXPECT_TRUE(isSorted(items, byKeyThenId));
    // The depth-limit fallback on its own: must give the identical (total-order) result.
    sort_detail::heapSort(viaHeap.data(), viaHeap.size(), byKeyThenId);
    for (std::size_t i = 0; i < items.size(); ++i) ASSERT_EQ(viaHeap[i].id, items[i].id);
}

TEST(FlatSet, InsertContainsAndOrder) {
    FlatSet<int> s;
    EXPECT_TRUE(s.insert(5));
    EXPECT_TRUE(s.insert(1));
    EXPECT_TRUE(s.insert(9));
    EXPECT_FALSE(s.insert(5));  // duplicate
    EXPECT_EQ(s.size(), 3u);
    EXPECT_TRUE(s.contains(9));
    EXPECT_FALSE(s.contains(4));
    EXPECT_EQ(s.count(1), 1u);
    EXPECT_EQ(s.count(2), 0u);
    Vector<int> order;
    for (int x : s) order.push_back(x);
    const Vector<int> expected{1, 5, 9};
    EXPECT_EQ(order, expected);
}

TEST(FlatMap, IndexFindAndOrder) {
    FlatMap<std::string, std::string> m;
    m["k"] = "1";
    m["ef"] = "80";
    m["k"] = "5";  // overwrite, no new entry
    EXPECT_EQ(m.size(), 2u);
    ASSERT_NE(m.find("k"), nullptr);
    EXPECT_EQ(*m.find("k"), "5");
    EXPECT_EQ(m.find("missing"), nullptr);
    EXPECT_TRUE(m.contains("ef"));
    EXPECT_EQ(m.begin()->key, "ef");  // sorted by key
}

TEST(FixedArray, AggregateAndSpan) {
    FixedArray<int, 4> a{1, 2, 3, 4};
    std::span<const int> s = a;
    EXPECT_EQ(s.size(), 4u);
    EXPECT_EQ(s[3], 4);
    FixedArray<int, 3> zero{};
    for (int x : zero) EXPECT_EQ(x, 0);
}

TEST(Basic, MinMaxEqual) {
    EXPECT_EQ(minOf(3, 7), 3);
    EXPECT_EQ(maxOf(3, 7), 7);
    const Vector<int> a{1, 2, 3}, b{1, 2, 3}, c{1, 2};
    EXPECT_TRUE(equalRanges<int>(a, b));
    EXPECT_FALSE(equalRanges<int>(a, c));
}
