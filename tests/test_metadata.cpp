#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "data/errors.hpp"
#include "data/metadata_manager.hpp"

using namespace sir;
namespace fs = std::filesystem;

namespace {
fs::path writeJson(const std::string& name, const std::string& body) {
    const auto dir = fs::temp_directory_path() / "sir_tests";
    fs::create_directories(dir);
    std::ofstream(dir / name) << body;
    return dir / name;
}
}  // namespace

TEST(Metadata, LoadsRecordsAndResolvesPaths) {
    const auto p = writeJson("meta.json", R"({"count": 2, "records": [
      {"internal_id": 0, "image_id": "100", "path": "img/100.jpg", "width": 500, "height": 375,
       "split": "train", "captions": ["a dog", "a cat"]},
      {"internal_id": 1, "image_id": "200", "path": "img/200.jpg", "width": 333, "height": 500,
       "split": "test", "captions": ["a car"]}]})");
    const auto m = MetadataManager::load(p, "/root");
    ASSERT_EQ(m.size(), 2u);
    EXPECT_EQ(m.at(1).imageId, "200");
    EXPECT_EQ(m.at(1).path, fs::path("/root/img/200.jpg"));
    EXPECT_EQ(m.at(0).captions.size(), 2u);
    EXPECT_THROW(m.at(2), DataError);
    EXPECT_THROW(m.at(-1), DataError);
}

TEST(Metadata, RejectsNonContiguousIdsAndBadCounts) {
    const auto gap = writeJson("gap.json", R"({"count": 1, "records": [
      {"internal_id": 1, "image_id": "x", "path": "x.jpg", "width": 1, "height": 1,
       "captions": ["c"]}]})");
    EXPECT_THROW(MetadataManager::load(gap, "/"), DataError);
    const auto cnt = writeJson("cnt.json", R"({"count": 5, "records": []})");
    EXPECT_THROW(MetadataManager::load(cnt, "/"), DataError);
    EXPECT_THROW(MetadataManager::load("/nonexistent/meta.json", "/"), DataError);
}

TEST(Metadata, RealFileIfPresent) {
    const fs::path real = fs::path(SIR_SOURCE_DIR) / "data/processed/metadata.json";
    if (!fs::exists(real)) GTEST_SKIP() << "metadata.json not generated";
    const auto m = MetadataManager::load(real, SIR_SOURCE_DIR);
    EXPECT_GT(m.size(), 0u);
    EXPECT_TRUE(fs::exists(m.at(0).path)) << m.at(0).path;
}
