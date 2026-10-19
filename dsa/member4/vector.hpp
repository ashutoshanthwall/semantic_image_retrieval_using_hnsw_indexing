#pragma once
// Vector<T>: a growable array owning one contiguous heap buffer [data, data + capacity).
// The first `size` slots hold constructed elements; the rest is raw storage.
//
//   push_back / emplace_back   amortised O(1): when full, capacity doubles (1, 2, 4, ...),
//                              so n pushes cost O(n) element moves in total
//   operator[]                 O(1), unchecked;  at() is the bounds-checked version
//   reserve(n)                 allocates exactly n if n > capacity (no growth afterwards
//                              until size exceeds n), which the indexes rely on for their
//                              memoryBytes() accounting
//   clear()                    destroys the elements but keeps the buffer
//
// Elements are relocated with their move constructor when it cannot throw, otherwise
// copied, so a throwing copy during growth leaves the vector unchanged. Iterators are raw
// pointers, so Vector converts implicitly to std::span and works in range-for loops.
// Any growth invalidates pointers into the buffer.
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sir {

template <typename T>
class Vector {
    static_assert(alignof(T) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__,
                  "Vector uses the default operator new alignment");

    // Largest element count whose byte size fits in an object (PTRDIFF_MAX bytes).
    static constexpr std::size_t kMaxSize =
        static_cast<std::size_t>(PTRDIFF_MAX) / sizeof(T);

public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    Vector() noexcept = default;
    explicit Vector(std::size_t n) { resize(n); }  // n value-initialised elements
    Vector(std::size_t n, const T& value) { assign(n, value); }
    Vector(const T* first, const T* last) {
        append(first, static_cast<std::size_t>(last - first));
    }
    Vector(std::initializer_list<T> items) : Vector(items.begin(), items.end()) {}
    Vector(const Vector& other) : Vector(other.begin(), other.end()) {}
    Vector(Vector&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = other.capacity_ = 0;
    }

    Vector& operator=(const Vector& other) {
        if (this != &other) {
            clear();
            if (other.size_ > capacity_) regrow(other.size_, 0, [](T*, std::size_t&) {});
            for (const T& x : other) {
                ::new (static_cast<void*>(data_ + size_)) T(x);
                ++size_;
            }
        }
        return *this;
    }
    Vector& operator=(Vector&& other) noexcept {
        if (this != &other) {
            destroy(data_, size_);
            deallocate(data_);
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = other.capacity_ = 0;
        }
        return *this;
    }
    ~Vector() {
        destroy(data_, size_);
        deallocate(data_);
    }

    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    T* begin() noexcept { return data_; }
    T* end() noexcept { return data_ + size_; }
    const T* begin() const noexcept { return data_; }
    const T* end() const noexcept { return data_ + size_; }

    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }
    T& at(std::size_t i) {
        if (i >= size_) throw std::out_of_range("Vector::at: index out of range");
        return data_[i];
    }
    const T& at(std::size_t i) const {
        if (i >= size_) throw std::out_of_range("Vector::at: index out of range");
        return data_[i];
    }
    T& front() noexcept { return data_[0]; }
    const T& front() const noexcept { return data_[0]; }
    T& back() noexcept { return data_[size_ - 1]; }
    const T& back() const noexcept { return data_[size_ - 1]; }

    void reserve(std::size_t n) {
        if (n > capacity_) regrow(n, 0, [](T*, std::size_t&) {});
    }

    void clear() noexcept {
        destroy(data_, size_);
        size_ = 0;
    }

    template <typename... Args>
    T& emplace_back(Args&&... args) {
        if (size_ == capacity_) {
            // The new element is built in the new buffer before the old one is released,
            // so `args` may safely refer to an element of this vector.
            regrow(grownCapacity(size_ + 1), 1, [&](T* slot, std::size_t& built) {
                ::new (static_cast<void*>(slot)) T(std::forward<Args>(args)...);
                built = 1;
            });
        } else {
            ::new (static_cast<void*>(data_ + size_)) T(std::forward<Args>(args)...);
            ++size_;
        }
        return data_[size_ - 1];
    }
    void push_back(const T& value) { emplace_back(value); }
    void push_back(T&& value) { emplace_back(std::move(value)); }
    void pop_back() noexcept {
        --size_;
        destroy(data_ + size_, 1);
    }

    // Appends copies of src[0, n). `src` may point into this vector.
    void append(const T* src, std::size_t n) {
        if (n == 0) return;
        if (n > capacity_ - size_) {
            if (n > kMaxSize - size_)
                throw std::length_error("Vector: size would exceed the maximum");
            regrow(grownCapacity(size_ + n), n, [&](T* slot, std::size_t& built) {
                for (; built < n; ++built) ::new (static_cast<void*>(slot + built)) T(src[built]);
            });
            return;
        }
        for (std::size_t i = 0; i < n; ++i) {
            ::new (static_cast<void*>(data_ + size_)) T(src[i]);
            ++size_;
        }
    }

    // Inserts before position `pos` (0 <= pos <= size), shifting later elements right. O(n).
    void insertAt(std::size_t pos, T value) {
        emplace_back(std::move(value));
        T moved = std::move(data_[size_ - 1]);
        for (std::size_t i = size_ - 1; i > pos; --i) data_[i] = std::move(data_[i - 1]);
        data_[pos] = std::move(moved);
    }

    void resize(std::size_t n) {
        resizeWith(n, [](T* p) { ::new (static_cast<void*>(p)) T(); });
    }
    void resize(std::size_t n, const T& value) {
        const T fill(value);  // `value` may be an element that growth would move away
        resizeWith(n, [&fill](T* p) { ::new (static_cast<void*>(p)) T(fill); });
    }

    // Replaces the contents with n copies of value. Allocates exactly n when growing.
    void assign(std::size_t n, const T& value) {
        const T fill(value);  // `value` may be an element that clear() destroys
        clear();
        if (n > capacity_) regrow(n, 0, [](T*, std::size_t&) {});
        for (; size_ < n; ++size_) ::new (static_cast<void*>(data_ + size_)) T(fill);
    }

private:
    static T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > kMaxSize) throw std::length_error("Vector: requested capacity is too large");
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }
    static void deallocate(T* p) noexcept { ::operator delete(p); }
    static void destroy(T* p, std::size_t n) noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>)
            for (std::size_t i = 0; i < n; ++i) p[i].~T();
    }

    // Doubling growth: the smallest capacity >= `required` that is at least twice the size.
    std::size_t grownCapacity(std::size_t required) const noexcept {
        const std::size_t doubled = size_ > kMaxSize / 2 ? kMaxSize : 2 * size_;
        return required > doubled ? required : doubled;
    }

    // Moves to a new buffer of `newCap` slots. `constructTail(slot, built)` constructs
    // `tailCount` new elements starting at `slot` (just past the existing ones), counting
    // them in `built`. Strong guarantee: if anything throws, *this is unchanged.
    template <typename ConstructTail>
    void regrow(std::size_t newCap, std::size_t tailCount, ConstructTail constructTail) {
        if (newCap < size_ || newCap - size_ < tailCount)  // unreachable; states the invariant
            throw std::logic_error("Vector: new capacity is below the new size");
        T* fresh = allocate(newCap);
        std::size_t built = 0;
        try {
            constructTail(fresh + size_, built);
            relocate(fresh);
        } catch (...) {
            destroy(fresh + size_, built);
            deallocate(fresh);
            throw;
        }
        destroy(data_, size_);
        deallocate(data_);
        data_ = fresh;
        capacity_ = newCap;
        size_ += tailCount;
    }

    // Move-constructs (or copy-constructs, if moving could throw) the elements into `to`.
    // If a copy throws, the copies made so far are destroyed and the old buffer is intact.
    void relocate(T* to) {
        if constexpr (std::is_nothrow_move_constructible_v<T>) {
            for (std::size_t i = 0; i < size_; ++i)
                ::new (static_cast<void*>(to + i)) T(std::move(data_[i]));
        } else {
            std::size_t i = 0;
            try {
                for (; i < size_; ++i) ::new (static_cast<void*>(to + i)) T(data_[i]);
            } catch (...) {
                destroy(to, i);
                throw;
            }
        }
    }

    template <typename ConstructOne>
    void resizeWith(std::size_t n, ConstructOne constructOne) {
        if (n <= size_) {
            destroy(data_ + n, size_ - n);
            size_ = n;
            return;
        }
        const std::size_t extra = n - size_;
        if (n > capacity_) {
            regrow(grownCapacity(n), extra, [&](T* slot, std::size_t& built) {
                for (; built < extra; ++built) constructOne(slot + built);
            });
            return;
        }
        for (; size_ < n; ++size_) constructOne(data_ + size_);
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

template <typename T>
bool operator==(const Vector<T>& a, const Vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (!(a[i] == b[i])) return false;
    return true;
}

}  // namespace sir
