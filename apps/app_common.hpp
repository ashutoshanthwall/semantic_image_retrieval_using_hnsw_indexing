#pragma once
// Small helpers shared by the executables: argument parsing and loading the standard
// artefacts (config, metadata, embeddings, query embeddings) with validation.
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "data/config.hpp"
#include "data/metadata_manager.hpp"
#include "embedding/embedding_store.hpp"

namespace sir::app {

struct Args {
    std::vector<std::string> positional;
    std::map<std::string, std::string> options;  // --key value  or  --flag (value "1")
    bool has(const std::string& k) const { return options.count(k) != 0; }
    std::string get(const std::string& k, const std::string& def) const {
        auto it = options.find(k);
        return it == options.end() ? def : it->second;
    }
    int getInt(const std::string& k, int def) const {
        auto it = options.find(k);
        return it == options.end() ? def : std::stoi(it->second);
    }
};

inline Args parseArgs(int argc, char** argv, const std::vector<std::string>& flags = {}) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) == 0) {
            const std::string key = s.substr(2);
            bool isFlag = false;
            for (const auto& f : flags) isFlag |= (f == key);
            if (isFlag || i + 1 >= argc) a.options[key] = "1";
            else a.options[key] = argv[++i];
        } else {
            a.positional.push_back(s);
        }
    }
    return a;
}

inline AppConfig loadAppConfig(const Args& a) {
    return loadConfig(a.get("config", "config/config.json"));
}

inline EmbeddingStore loadImageStore(const AppConfig& c) {
    return EmbeddingStore::load(c.embeddings, c.manifest, c.dim, c.modelName, c.normTolerance);
}

inline EmbeddingStore loadQueryStore(const AppConfig& c) {
    return EmbeddingStore::load(c.queryEmbeddings, c.queryManifest, c.dim, c.modelName,
                                c.normTolerance);
}

}  // namespace sir::app
