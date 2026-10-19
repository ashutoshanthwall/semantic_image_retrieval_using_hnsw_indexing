#include <gtest/gtest.h>

#include <filesystem>
#include <random>
#include <string>

#include "algorithms/basic.hpp"
#include "containers/hash_map.hpp"
#include "data/errors.hpp"
#include "data/inverted_index.hpp"

using namespace sir;
namespace fs = std::filesystem;

namespace {

// Every key lands in slot 0, so each lookup has to walk the whole cluster.
struct ConstantHash {
    std::uint64_t operator()(int) const noexcept { return 0; }
};

Vector<std::string> words(std::initializer_list<const char*> ws) {
    Vector<std::string> out;
    for (const char* w : ws) out.push_back(w);
    return out;
}

bool contains(const Vector<std::string>& ws, const std::string& w) {
    for (const auto& x : ws)
        if (x == w) return true;
    return false;
}

// Reference answers by scanning every document's words.
Vector<int> scanAll(const Vector<Vector<std::string>>& docs, const Vector<std::string>& q) {
    Vector<int> out;
    if (q.empty()) return out;
    for (std::size_t d = 0; d < docs.size(); ++d) {
        bool all = true;
        for (const auto& w : q) all = all && contains(docs[d], w);
        if (all) out.push_back(static_cast<int>(d));
    }
    return out;
}
Vector<int> scanAny(const Vector<Vector<std::string>>& docs, const Vector<std::string>& q) {
    Vector<int> out;
    for (std::size_t d = 0; d < docs.size(); ++d) {
        bool any = false;
        for (const auto& w : q) any = any || contains(docs[d], w);
        if (any) out.push_back(static_cast<int>(d));
    }
    return out;
}

std::string join(const Vector<std::string>& ws) {
    std::string s;
    for (const auto& w : ws) s += w + " ";
    return s;
}

}  // namespace

TEST(HashMap, InsertFindAndGrow) {
    HashMap<std::string, int> m;
    EXPECT_EQ(m.find("x"), nullptr);
    for (int i = 0; i < 1000; ++i) m["k" + std::to_string(i)] = i;
    ASSERT_EQ(m.size(), 1000u);
    for (int i = 0; i < 1000; ++i) {
        const int* v = m.find("k" + std::to_string(i));
        ASSERT_NE(v, nullptr);
        EXPECT_EQ(*v, i);
    }
    EXPECT_FALSE(m.contains("k1000"));
    m["k5"] += 10;  // existing key: no new entry
    EXPECT_EQ(m.size(), 1000u);
    EXPECT_EQ(*m.find("k5"), 15);
    // Capacity is a power of two and the load factor never exceeds 0.7.
    EXPECT_EQ(m.capacity() & (m.capacity() - 1), 0u);
    EXPECT_LE(m.loadFactor(), 0.7);
    EXPECT_GE(m.averageProbeLength(), 1.0);
}

TEST(HashMap, CorrectUnderTotalCollision) {
    HashMap<int, int, ConstantHash> m;
    for (int i = 0; i < 200; ++i) m[i] = i * i;
    for (int i = 0; i < 200; ++i) ASSERT_EQ(*m.find(i), i * i);
    EXPECT_EQ(m.find(200), nullptr);
    // One cluster: the key at distance d from slot 0 needs d + 1 probes, mean (n + 1) / 2.
    EXPECT_DOUBLE_EQ(m.averageProbeLength(), 100.5);
}

TEST(HashMap, ForEachVisitsEveryEntryOnce) {
    HashMap<int, int> m;
    for (int i = 0; i < 300; ++i) m[i] = 1;
    Vector<int> seen(300, 0);
    m.forEach([&](int k, int v) { seen[static_cast<std::size_t>(k)] += v; });
    for (int c : seen) EXPECT_EQ(c, 1);
}

TEST(InvertedIndex, Tokenize) {
    EXPECT_EQ(tokenize("A Dog, in the SNOW!"), words({"dog", "snow"}));
    EXPECT_EQ(tokenize("two-year-old  child's 3 toys"),
              words({"two", "year", "old", "child", "s", "3", "toys"}));
    EXPECT_EQ(tokenize("caf\xc3\xa9 au lait"), words({"caf\xc3\xa9", "au", "lait"}));
    EXPECT_TRUE(tokenize("the a of ...").empty());
}

TEST(InvertedIndex, PostingsAreSortedAndUnique) {
    InvertedIndex idx;
    idx.add(0, "a dog runs");
    idx.add(0, "the dog jumps");  // second caption of image 0: no duplicate posting
    idx.add(2, "a cat");
    idx.add(5, "dog and cat");
    EXPECT_TRUE(equalRanges(idx.postings("dog"), std::span<const int>(Vector<int>{0, 5})));
    EXPECT_TRUE(equalRanges(idx.postings("cat"), std::span<const int>(Vector<int>{2, 5})));
    EXPECT_TRUE(idx.postings("bird").empty());
    EXPECT_EQ(idx.postingCount(), 6u);  // dog x2, runs, jumps, cat x2 ("and" is a stop word)
    EXPECT_THROW(idx.add(4, "late"), DataError);
}

TEST(InvertedIndex, BooleanQueries) {
    InvertedIndex idx;
    idx.add(0, "dog snow");
    idx.add(1, "dog beach");
    idx.add(2, "cat snow");
    idx.add(3, "dog cat snow");
    EXPECT_EQ(idx.matchAll("dog snow"), (Vector<int>{0, 3}));
    EXPECT_EQ(idx.matchAll("Snow, DOG, snow"), (Vector<int>{0, 3}));
    EXPECT_TRUE(idx.matchAll("dog bird").empty());  // unknown word empties an AND
    EXPECT_TRUE(idx.matchAll("the").empty());       // nothing left after stop words
    EXPECT_EQ(idx.matchAny("beach cat"), (Vector<int>{1, 2, 3}));
    EXPECT_EQ(idx.matchAny("bird beach"), (Vector<int>{1}));
    EXPECT_TRUE(idx.matchAny("").empty());
}

TEST(InvertedIndex, IntersectionsAgree) {
    std::mt19937 rng(7);
    for (int trial = 0; trial < 200; ++trial) {
        Vector<int> a, b;
        for (int x = 0; x < 5000; ++x) {
            if (rng() % 40 == 0) a.push_back(x);
            if (rng() % (1 + trial % 5) == 0) b.push_back(x);
        }
        EXPECT_EQ(intersectMerge(a, b), intersectGallop(a, b));
    }
    EXPECT_TRUE(intersectGallop(Vector<int>{}, Vector<int>{1, 2}).empty());
    EXPECT_EQ(intersectGallop(Vector<int>{9}, Vector<int>{1, 9}), (Vector<int>{9}));
}

// Random documents over a small vocabulary, so queries have overlapping, uneven lists.
TEST(InvertedIndex, MatchesLinearScan) {
    const Vector<std::string> vocab =
        words({"dog", "cat", "man", "woman", "red", "ball", "snow", "beach", "car", "child"});
    std::mt19937 rng(11);
    Vector<Vector<std::string>> docs;
    InvertedIndex idx;
    for (int d = 0; d < 2000; ++d) {
        Vector<std::string> doc;
        const int len = 1 + static_cast<int>(rng() % 6);
        // Skewed choice: low-numbered words are far more common, like real captions.
        for (int t = 0; t < len; ++t)
            doc.push_back(vocab[minOf<std::size_t>(rng() % 4 * (rng() % 4), vocab.size() - 1)]);
        idx.add(d, join(doc));
        docs.push_back(std::move(doc));
    }
    for (int q = 0; q < 300; ++q) {
        Vector<std::string> query;
        const int len = 1 + static_cast<int>(rng() % 3);
        for (int t = 0; t < len; ++t) query.push_back(vocab[rng() % vocab.size()]);
        EXPECT_EQ(idx.matchAll(join(query)), scanAll(docs, query)) << join(query);
        EXPECT_EQ(idx.matchAny(join(query)), scanAny(docs, query)) << join(query);
    }
}

TEST(InvertedIndex, BuildSkipsHeldOutCaption) {
    Vector<ImageRecord> recs(2);
    recs[0].internalId = 0;
    recs[0].captions = words({"held out zebra", "a dog"});
    recs[1].internalId = 1;
    recs[1].captions = words({"a cat", "zebra crossing"});
    const MetadataManager meta(std::move(recs));
    const InvertedIndex idx = InvertedIndex::build(meta, 0);
    EXPECT_EQ(idx.matchAll("zebra"), (Vector<int>{1}));  // image 0's zebra was caption 0
    EXPECT_TRUE(idx.matchAll("cat").empty());
    EXPECT_EQ(InvertedIndex::build(meta, -1).matchAll("zebra"), (Vector<int>{0, 1}));
}

TEST(InvertedIndex, RealMetadataIfPresent) {
    const fs::path real = fs::path(SIR_SOURCE_DIR) / "data/processed/metadata.json";
    if (!fs::exists(real)) GTEST_SKIP() << "metadata.json not generated";
    const auto meta = MetadataManager::load(real, SIR_SOURCE_DIR);
    const InvertedIndex idx = InvertedIndex::build(meta, 0);
    EXPECT_GT(idx.vocabularySize(), 1000u);
    const auto dogs = idx.postings("dog");
    ASSERT_FALSE(dogs.empty());
    for (std::size_t i = 1; i < dogs.size(); ++i) ASSERT_LT(dogs[i - 1], dogs[i]);
}
