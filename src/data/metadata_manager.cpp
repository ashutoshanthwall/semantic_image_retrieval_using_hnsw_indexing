#include "data/metadata_manager.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "data/errors.hpp"

namespace sir {

MetadataManager::MetadataManager(Vector<ImageRecord> records)
    : records_(std::move(records)) {
    for (std::size_t i = 0; i < records_.size(); ++i)
        if (records_[i].internalId != static_cast<int>(i))
            throw DataError("metadata ids are not contiguous: record " + std::to_string(i) +
                            " has internal_id " + std::to_string(records_[i].internalId));
}

MetadataManager MetadataManager::load(const std::filesystem::path& metadataJson,
                                      const std::filesystem::path& root) {
    std::ifstream in(metadataJson);
    if (!in) throw DataError("cannot open metadata file: " + metadataJson.string());
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(in);
        Vector<ImageRecord> recs;
        const auto& arr = j.at("records");
        recs.reserve(arr.size());
        for (const auto& r : arr) {
            ImageRecord rec;
            rec.internalId = r.at("internal_id").get<int>();
            rec.imageId = r.at("image_id").get<std::string>();
            rec.path = root / r.at("path").get<std::string>();
            rec.width = r.at("width").get<int>();
            rec.height = r.at("height").get<int>();
            rec.split = r.value("split", "");
            const auto& caps = r.at("captions");
            if (!caps.is_array())
                throw DataError("captions of record " + std::to_string(rec.internalId) +
                                " is not an array");
            rec.captions.reserve(caps.size());
            for (const auto& c : caps) rec.captions.push_back(c.get<std::string>());
            recs.push_back(std::move(rec));
        }
        if (j.at("count").get<std::size_t>() != recs.size())
            throw DataError("metadata count field disagrees with number of records");
        return MetadataManager(std::move(recs));
    } catch (const nlohmann::json::exception& e) {
        throw DataError("malformed metadata " + metadataJson.string() + ": " + e.what());
    }
}

const ImageRecord& MetadataManager::at(int internalId) const {
    if (internalId < 0 || static_cast<std::size_t>(internalId) >= records_.size())
        throw DataError("internal id " + std::to_string(internalId) + " out of range [0, " +
                        std::to_string(records_.size()) + ")");
    return records_[static_cast<std::size_t>(internalId)];
}

}  // namespace sir
