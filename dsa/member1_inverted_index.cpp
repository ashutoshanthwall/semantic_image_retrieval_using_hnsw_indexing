// Member 1: caption search with a hash table and an inverted index.
//
// HashTable  - open addressing with linear probing. All slots are in one array whose
//              length is a power of two, so a key's home slot is hash(key) & (capacity - 1).
//              On a collision the key goes in the next free slot (wrapping around), and a
//              lookup walks the same path until it finds the key or an empty slot.
//              Expected O(1) per lookup while the table is at most 70% full; when an insert
//              would pass 70%, the table doubles and every key is inserted again (rehash).
//              There is no delete: emptying a slot would cut the probe path of the keys
//              stored after it, and the inverted index never removes words.
//
// InvertedIndex - for every caption word, the sorted list of image ids containing it.
//   build:     one pass over the captions. Images are added in id order, so each list is
//              built already sorted: an id is appended only if it differs from the last one.
//   matchAll:  AND query. Intersect the lists shortest first, so the running result never
//              grows. Each step is a two-pointer merge, O(a + b), or a galloping search,
//              O(a log(b / a)), when one list is much shorter than the other.
//   matchAny:  OR query. Repeated two-pointer union of the lists.
#include "data.h"

// ---------------------------------------------------------------- hash table

// FNV-1a, 64-bit: for each byte, xor it into the hash, then multiply by the FNV prime.
uint64_t fnv1aHash(const std::string& key) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < key.size(); i++) {
        h ^= (unsigned char)key[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

HashTable::HashTable(HashFunction function) {
    slots = nullptr;
    numSlots = 0;
    count = 0;
    hash = function;
}

HashTable::~HashTable() {
    delete[] slots;
}

int HashTable::homeSlot(const std::string& key) const {
    return (int)(hash(key) & (uint64_t)(numSlots - 1));
}

// Doubles the slot array (first size 8) and puts every key in its slot in the new array.
void HashTable::grow() {
    Slot* old = slots;
    int oldSize = numSlots;
    numSlots = oldSize == 0 ? 8 : 2 * oldSize;
    slots = new Slot[numSlots];
    for (int i = 0; i < oldSize; i++) {
        if (!old[i].used) continue;
        int s = homeSlot(old[i].key);
        while (slots[s].used) s = (s + 1) & (numSlots - 1);
        slots[s] = old[i];
    }
    delete[] old;
}

IntArray& HashTable::getOrInsert(const std::string& key) {
    if (10 * (count + 1) > 7 * numSlots) grow();  // keep the load factor <= 0.7
    int s = homeSlot(key);
    while (slots[s].used) {
        if (slots[s].key == key) return slots[s].value;
        s = (s + 1) & (numSlots - 1);
    }
    slots[s].used = true;
    slots[s].key = key;
    count++;
    return slots[s].value;
}

const IntArray* HashTable::find(const std::string& key) const {
    if (numSlots == 0) return nullptr;
    int s = homeSlot(key);
    while (slots[s].used) {
        if (slots[s].key == key) return &slots[s].value;
        s = (s + 1) & (numSlots - 1);
    }
    return nullptr;
}

int HashTable::size() const { return count; }
int HashTable::capacity() const { return numSlots; }

double HashTable::loadFactor() const {
    return numSlots == 0 ? 0.0 : (double)count / numSlots;
}

// A key at distance d from its home slot needs d + 1 probes to be found. Linear-probing
// theory predicts about (1 + 1 / (1 - a)) / 2 on average at load factor a.
double HashTable::averageProbeLength() const {
    if (count == 0) return 0.0;
    long long total = 0;
    for (int i = 0; i < numSlots; i++)
        if (slots[i].used) total += ((i - homeSlot(slots[i].key)) & (numSlots - 1)) + 1;
    return (double)total / count;
}

// ---------------------------------------------------------------- tokenizer

static const char* STOP_WORDS[] = {"a",  "an", "and", "are", "at", "for", "in",
                                   "is", "of", "on",  "the", "to", "with"};
static const int NUM_STOP_WORDS = 13;

static bool isStopWord(const std::string& word) {
    for (int i = 0; i < NUM_STOP_WORDS; i++)
        if (word == STOP_WORDS[i]) return true;
    return false;
}

// Letters, digits, and bytes >= 0x80 (so UTF-8 words like "café" stay whole).
static bool isWordByte(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c >= 0x80;
}

void tokenize(const std::string& text, StringArray& words) {
    words.clear();
    std::string word;
    for (size_t i = 0; i <= text.size(); i++) {
        // One step past the end acts as a separator, which ends the last word.
        unsigned char c = i < text.size() ? (unsigned char)text[i] : ' ';
        if (isWordByte(c)) {
            if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
            word += (char)c;
        } else if (!word.empty()) {
            if (!isStopWord(word)) words.push(word);
            word.clear();
        }
    }
}

// ---------------------------------------------------------------- sorted-list operations

void intersectMerge(const IntArray& a, const IntArray& b, IntArray& out) {
    out.clear();
    int i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) {
            i++;
        } else if (b[j] < a[i]) {
            j++;
        } else {
            out.push(a[i]);
            i++;
            j++;
        }
    }
}

// First position in list[from..] whose value is >= x. Takes steps of 1, 2, 4, ... until it
// passes x, then binary searches inside the last step.
static int gallop(const IntArray& list, int from, int x) {
    int low = from, high = from, step = 1;
    while (high < list.size() && list[high] < x) {
        low = high + 1;
        high += step;
        step *= 2;
    }
    if (high > list.size()) high = list.size();
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (list[mid] < x)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

void intersectGallop(const IntArray& small, const IntArray& large, IntArray& out) {
    out.clear();
    int pos = 0;
    for (int i = 0; i < small.size(); i++) {
        pos = gallop(large, pos, small[i]);  // ids only grow, so the search resumes at pos
        if (pos == large.size()) break;
        if (large[pos] == small[i]) out.push(small[i]);
    }
}

void unionMerge(const IntArray& a, const IntArray& b, IntArray& out) {
    out.clear();
    int i = 0, j = 0;
    while (i < a.size() || j < b.size()) {
        if (j == b.size() || (i < a.size() && a[i] < b[j])) {
            out.push(a[i]);
            i++;
        } else if (i == a.size() || b[j] < a[i]) {
            out.push(b[j]);
            j++;
        } else {  // same id in both lists: keep it once
            out.push(a[i]);
            i++;
            j++;
        }
    }
}

// ---------------------------------------------------------------- inverted index

InvertedIndex::InvertedIndex() {
    totalPostings = 0;
    lastId = -1;
}

void InvertedIndex::build(const MetadataManager& meta, int skipCaption) {
    for (int i = 0; i < meta.size(); i++) {
        const ImageRecord& r = meta.at(i);
        for (int c = 0; c < r.captions.size(); c++)
            if (c != skipCaption) add(i, r.captions[c]);
    }
}

void InvertedIndex::add(int id, const std::string& text) {
    if (id < lastId)
        throw Error("inverted index: id " + std::to_string(id) + " added after id " +
                    std::to_string(lastId) + " (ids must be added in order)");
    lastId = id;
    StringArray words;
    tokenize(text, words);
    for (int w = 0; w < words.size(); w++) {
        IntArray& list = terms.getOrInsert(words[w]);
        if (list.size() == 0 || list.back() != id) {
            list.push(id);
            totalPostings++;
        }
    }
}

const IntArray* InvertedIndex::postings(const std::string& term) const {
    return terms.find(term);
}

// Collects the posting list of each distinct query word into `lists` and returns false if
// some word is not in the index (which makes an AND query empty).
static bool queryLists(const InvertedIndex& index, const std::string& query,
                       const IntArray** lists, int& numLists) {
    StringArray words;
    tokenize(query, words);
    bool allFound = true;
    numLists = 0;
    for (int w = 0; w < words.size(); w++) {
        bool repeated = false;  // "dog and dog" is just "dog"
        for (int v = 0; v < w; v++)
            if (words[v] == words[w]) repeated = true;
        if (repeated) continue;
        const IntArray* list = index.postings(words[w]);
        if (list == nullptr)
            allFound = false;
        else
            lists[numLists++] = list;
    }
    return allFound;
}

static int countWords(const std::string& query) {
    StringArray words;
    tokenize(query, words);
    return words.size();
}

void InvertedIndex::matchAll(const std::string& query, IntArray& out) const {
    out.clear();
    const IntArray** lists = new const IntArray*[countWords(query) + 1];
    int n = 0;
    bool allFound = queryLists(*this, query, lists, n);
    if (allFound && n > 0) {
        // Insertion sort by length, so the shortest list comes first (a query has few words).
        for (int i = 1; i < n; i++) {
            const IntArray* item = lists[i];
            int j = i;
            while (j > 0 && lists[j - 1]->size() > item->size()) {
                lists[j] = lists[j - 1];
                j--;
            }
            lists[j] = item;
        }
        out = *lists[0];
        IntArray next;
        for (int i = 1; i < n && out.size() > 0; i++) {
            // Galloping wins when the other list is much longer: it jumps over runs of ids
            // instead of stepping through each one.
            if (lists[i]->size() > 8 * out.size())
                intersectGallop(out, *lists[i], next);
            else
                intersectMerge(out, *lists[i], next);
            out = next;
        }
    }
    delete[] lists;
}

void InvertedIndex::matchAny(const std::string& query, IntArray& out) const {
    out.clear();
    const IntArray** lists = new const IntArray*[countWords(query) + 1];
    int n = 0;
    queryLists(*this, query, lists, n);
    IntArray merged;
    for (int i = 0; i < n; i++) {
        unionMerge(out, *lists[i], merged);
        out = merged;
    }
    delete[] lists;
}

int InvertedIndex::vocabularySize() const { return terms.size(); }
long long InvertedIndex::postingCount() const { return totalPostings; }
const HashTable& InvertedIndex::table() const { return terms; }
