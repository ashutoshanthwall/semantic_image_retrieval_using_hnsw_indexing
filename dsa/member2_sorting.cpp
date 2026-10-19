// Member 2: ordering of search results - comparison, introsort, binary heap and top-K.
//
// Every ordering in the engine uses closer(): by distance, then by smaller id. Ids are
// unique, so this is a total order and results never depend on insertion order or on
// which sorting algorithm ran.
//
// sortNeighbors is an introsort:
//   1. quicksort with a median-of-three pivot and Hoare partitioning; it recurses into the
//      smaller part and loops on the larger, so the recursion depth is O(log n);
//   2. if the recursion goes deeper than 2*log2(n) (a run of bad pivots), that part is
//      finished with heap sort, so the worst case stays O(n log n);
//   3. parts of 16 or fewer items are left for one insertion-sort pass at the end; each item
//      then moves at most 15 places, so that pass is O(n).
#include "search.h"

bool closer(const Neighbor& a, const Neighbor& b) {
    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
}

static const int INSERTION_THRESHOLD = 16;

static void swapItems(Neighbor& a, Neighbor& b) {
    Neighbor temp = a;
    a = b;
    b = temp;
}

static void insertionSort(Neighbor* a, int n) {
    for (int i = 1; i < n; i++) {
        Neighbor item = a[i];
        int j = i;
        while (j > 0 && closer(item, a[j - 1])) {
            a[j] = a[j - 1];
            j--;
        }
        a[j] = item;
    }
}

// Max-heap sift-down for heap sort: the farthest item rises to the top.
static void heapSiftDown(Neighbor* a, int n, int i) {
    Neighbor item = a[i];
    while (true) {
        int child = 2 * i + 1;
        if (child >= n) break;
        if (child + 1 < n && closer(a[child], a[child + 1])) child++;  // the farther child
        if (!closer(item, a[child])) break;
        a[i] = a[child];
        i = child;
    }
    a[i] = item;
}

static void heapSort(Neighbor* a, int n) {
    for (int i = n / 2 - 1; i >= 0; i--) heapSiftDown(a, n, i);  // build a max-heap
    for (int end = n - 1; end > 0; end--) {
        swapItems(a[0], a[end]);  // farthest item goes to the back
        heapSiftDown(a, end, 0);
    }
}

// Hoare partition of a[0..n-1], n >= 3. Returns s with 0 < s < n such that every item of
// a[0..s-1] is <= every item of a[s..n-1].
static int partition(Neighbor* a, int n) {
    // Median of three: order a[0] <= a[mid] <= a[n-1]. The two outer items then stop both
    // scans before they can run off either end.
    int mid = n / 2, last = n - 1;
    if (closer(a[mid], a[0])) swapItems(a[mid], a[0]);
    if (closer(a[last], a[mid])) {
        swapItems(a[last], a[mid]);
        if (closer(a[mid], a[0])) swapItems(a[mid], a[0]);
    }
    Neighbor pivot = a[mid];
    int i = -1, j = n;
    while (true) {
        do i++; while (closer(a[i], pivot));
        do j--; while (closer(pivot, a[j]));
        if (i >= j) return j + 1;
        swapItems(a[i], a[j]);
    }
}

static void introsortLoop(Neighbor* a, int n, int depthLeft) {
    while (n > INSERTION_THRESHOLD) {
        if (depthLeft == 0) {
            heapSort(a, n);
            return;
        }
        depthLeft--;
        int s = partition(a, n);
        if (s < n - s) {  // recurse into the smaller part, loop on the larger
            introsortLoop(a, s, depthLeft);
            a += s;
            n -= s;
        } else {
            introsortLoop(a + s, n - s, depthLeft);
            n = s;
        }
    }
}

void sortNeighbors(Neighbor* items, int n) {
    if (n < 2) return;
    int log2n = 0;
    for (int m = n; m > 1; m /= 2) log2n++;
    introsortLoop(items, n, 2 * log2n);
    insertionSort(items, n);
}

// ---------------------------------------------------------------- NeighborHeap

NeighborHeap::NeighborHeap(bool closestFirst) {
    closestOnTop = closestFirst;
}

bool NeighborHeap::ranksAbove(const Neighbor& a, const Neighbor& b) const {
    return closestOnTop ? closer(a, b) : closer(b, a);
}

// The item at slot i moves up while it ranks above its parent.
void NeighborHeap::siftUp(int i) {
    Neighbor item = items[i];
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (!ranksAbove(item, items[parent])) break;
        items[i] = items[parent];  // parent moves down into the gap
        i = parent;
    }
    items[i] = item;
}

// The item at slot i moves down while a child ranks above it.
void NeighborHeap::siftDown(int i) {
    int n = items.size();
    Neighbor item = items[i];
    while (true) {
        int child = 2 * i + 1;
        if (child >= n) break;
        if (child + 1 < n && ranksAbove(items[child + 1], items[child])) child++;
        if (!ranksAbove(items[child], item)) break;
        items[i] = items[child];  // child moves up into the gap
        i = child;
    }
    items[i] = item;
}

void NeighborHeap::push(const Neighbor& item) {
    items.push(item);
    siftUp(items.size() - 1);
}

void NeighborHeap::pop() {
    items[0] = items[items.size() - 1];  // last item fills the gap at the top
    items.pop();
    if (items.size() > 0) siftDown(0);
}

const Neighbor& NeighborHeap::top() const { return items[0]; }
int NeighborHeap::size() const { return items.size(); }
bool NeighborHeap::empty() const { return items.size() == 0; }
void NeighborHeap::reserve(int n) { items.reserve(n); }

// ---------------------------------------------------------------- TopK

TopK::TopK(int count) : heap(false) {
    k = count;
    heap.reserve(k + 1);
}

void TopK::push(const Neighbor& item) {
    if (k <= 0) return;
    if (heap.size() < k) {
        heap.push(item);
    } else if (closer(item, heap.top())) {  // beats the worst item kept
        heap.pop();
        heap.push(item);
    }
}

int TopK::size() const { return heap.size(); }

void TopK::extractSorted(NeighborArray& out) {
    int n = heap.size();
    out.clear();
    out.reserve(n);
    for (int i = 0; i < n; i++) out.push(Neighbor());
    for (int i = n - 1; i >= 0; i--) {  // the max-heap gives the farthest first
        out[i] = heap.top();
        heap.pop();
    }
}
