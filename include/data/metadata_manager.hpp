#pragma once
// MetadataManager owns the per-image records. Search works on contiguous integer ids
// 0..N-1 throughout; strings (image id, path, captions) are looked up only when a result
// is displayed.
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace sir {

struct ImageRecord {
    int internalId = -1;
    std::string imageId;
    std::filesystem::path path;  // absolute (resolved against the project root)
    int width = 0;
    int height = 0;
    std::string split;
    std::vector<std::string> captions;
};

class MetadataManager {
public:
    static MetadataManager load(const std::filesystem::path& metadataJson,
                                const std::filesystem::path& root);
    explicit MetadataManager(std::vector<ImageRecord> records);

    const ImageRecord& at(int internalId) const;  // throws DataError when out of range
    std::size_t size() const noexcept { return records_.size(); }

private:
    std::vector<ImageRecord> records_;
};

}  // namespace sir
