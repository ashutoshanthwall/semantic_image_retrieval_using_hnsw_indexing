#include "embedding/manifest.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "data/errors.hpp"

namespace sir {

EmbeddingManifest loadManifest(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw EmbeddingError("cannot open embedding manifest: " + path.string());
    try {
        const auto j = nlohmann::json::parse(in);
        EmbeddingManifest m;
        m.model = j.at("model").get<std::string>();
        m.dim = j.at("dim").get<std::size_t>();
        m.rows = j.at("rows").get<std::size_t>();
        m.l2Normalized = j.at("l2_normalized").get<bool>();
        m.dtype = j.at("dtype").get<std::string>();
        m.sha256 = j.at("sha256").get<std::string>();
        m.createdUtc = j.value("created_utc", "");
        m.gitCommit = j.value("git_commit", "");
        return m;
    } catch (const nlohmann::json::exception& e) {
        throw EmbeddingError("malformed manifest " + path.string() + ": " + e.what());
    }
}

}  // namespace sir
