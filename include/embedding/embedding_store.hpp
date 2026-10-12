#pragma once
// EmbeddingStore owns every embedding vector exactly once, as one contiguous row-major
// float32 buffer [N x D]. Row i is internal ID i. Indexes never copy vectors; they hold
// ids and read rows through row(i), which returns a non-owning std::span.
#include <cstddef>
#include <filesystem>
#include <span>
#include <string>

#include "containers/vector.hpp"
#include "embedding/manifest.hpp"

namespace sir {

class EmbeddingStore {
public:
    // Takes ownership of `data` (size must be a multiple of dim) and validates every row.
    EmbeddingStore(std::size_t dim, Vector<float> data, double normTolerance,
                   std::string sourceSha256 = {});

    // Reads a raw little-endian float32 file and validates it against its manifest and the
    // expected (config) dimension and model: size, dim, row count, dtype, normalised flag,
    // SHA-256, and per-row NaN/Inf/zero-norm/unit-norm checks.
    static EmbeddingStore load(const std::filesystem::path& binPath,
                               const std::filesystem::path& manifestPath,
                               std::size_t expectedDim, const std::string& expectedModel,
                               double normTolerance);

    EmbeddingStore(EmbeddingStore&&) noexcept = default;
    EmbeddingStore& operator=(EmbeddingStore&&) noexcept = default;
    EmbeddingStore(const EmbeddingStore&) = delete;  // the store is never copied implicitly
    EmbeddingStore& operator=(const EmbeddingStore&) = delete;

    std::span<const float> row(std::size_t i) const noexcept {
        return {data_.data() + i * dim_, dim_};
    }
    std::size_t size() const noexcept { return rows_; }
    std::size_t dim() const noexcept { return dim_; }
    std::size_t memoryBytes() const noexcept { return data_.capacity() * sizeof(float); }
    const std::string& sourceSha256() const noexcept { return sha256_; }

private:
    void validate(double normTolerance) const;

    std::size_t dim_ = 0;
    std::size_t rows_ = 0;
    Vector<float> data_;
    std::string sha256_;
};

// Validation helper shared with query vectors: throws EmbeddingError describing the
// first problem (wrong dim, NaN, Inf, zero norm, norm outside tolerance).
void validateVector(std::span<const float> v, std::size_t expectedDim, double normTolerance,
                    const std::string& what);

}  // namespace sir
