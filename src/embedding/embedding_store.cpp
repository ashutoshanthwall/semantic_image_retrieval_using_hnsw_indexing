#include "embedding/embedding_store.hpp"

#include <bit>
#include <cmath>
#include <fstream>
#include <sstream>

#include "data/errors.hpp"
#include "persistence/sha256.hpp"
#include "similarity/vector_math.hpp"

namespace sir {

static_assert(std::endian::native == std::endian::little,
              "embedding files are little-endian float32; add byte swapping for this host");

void validateVector(std::span<const float> v, std::size_t expectedDim, double normTolerance,
                    const std::string& what) {
    if (v.size() != expectedDim)
        throw EmbeddingError(what + ": dimension mismatch (got " + std::to_string(v.size()) +
                             ", expected " + std::to_string(expectedDim) + ")");
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (std::isnan(v[i]))
            throw EmbeddingError(what + ": NaN at element " + std::to_string(i));
        if (std::isinf(v[i]))
            throw EmbeddingError(what + ": Inf at element " + std::to_string(i));
    }
    const double n = norm(v);
    if (n == 0.0) throw EmbeddingError(what + ": zero-norm vector");
    if (std::abs(n - 1.0) > normTolerance) {
        std::ostringstream os;
        os.precision(9);
        os << what << ": L2 norm " << n << " outside 1 +- " << normTolerance
           << " (vectors must be unit-normalised)";
        throw EmbeddingError(os.str());
    }
}

EmbeddingStore::EmbeddingStore(std::size_t dim, std::vector<float> data, double normTolerance,
                               std::string sourceSha256)
    : dim_(dim), data_(std::move(data)), sha256_(std::move(sourceSha256)) {
    if (dim_ == 0) throw EmbeddingError("embedding dimension must be > 0");
    if (data_.size() % dim_ != 0)
        throw EmbeddingError("embedding buffer of " + std::to_string(data_.size()) +
                             " floats is not a multiple of dim " + std::to_string(dim_));
    rows_ = data_.size() / dim_;
    validate(normTolerance);
}

void EmbeddingStore::validate(double normTolerance) const {
    for (std::size_t i = 0; i < rows_; ++i)
        validateVector(row(i), dim_, normTolerance, "embedding row " + std::to_string(i));
}

EmbeddingStore EmbeddingStore::load(const std::filesystem::path& binPath,
                                    const std::filesystem::path& manifestPath,
                                    std::size_t expectedDim, const std::string& expectedModel,
                                    double normTolerance) {
    const EmbeddingManifest m = loadManifest(manifestPath);
    if (m.dim != expectedDim)
        throw EmbeddingError("dimension mismatch: manifest says " + std::to_string(m.dim) +
                             ", config says " + std::to_string(expectedDim));
    if (m.model != expectedModel)
        throw EmbeddingError("model mismatch: manifest '" + m.model + "', config '" +
                             expectedModel + "'");
    if (m.dtype != "float32") throw EmbeddingError("unsupported dtype " + m.dtype);
    if (!m.l2Normalized) throw EmbeddingError("manifest says embeddings are not L2-normalised");

    std::error_code ec;
    const auto bytes = std::filesystem::file_size(binPath, ec);
    if (ec) throw EmbeddingError("cannot stat embedding file " + binPath.string());
    const std::uintmax_t expected = std::uintmax_t{m.rows} * m.dim * sizeof(float);
    if (bytes != expected)
        throw EmbeddingError(binPath.string() + ": size " + std::to_string(bytes) +
                             " bytes, manifest implies " + std::to_string(expected) +
                             " (truncated or wrong file)");

    std::vector<float> data(m.rows * m.dim);
    std::ifstream in(binPath, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes)))
        throw EmbeddingError("short read from " + binPath.string());
    const std::string sha = sha256Hex(data.data(), bytes);
    if (sha != m.sha256)
        throw EmbeddingError(binPath.string() + ": SHA-256 " + sha +
                             " does not match manifest " + m.sha256);
    return EmbeddingStore(m.dim, std::move(data), normTolerance, sha);
}

}  // namespace sir
