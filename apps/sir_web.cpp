// Minimal local web UI: a search box and both backends' results side by side.
//
//   sir_web [--config f] [--port 8080] [--host 127.0.0.1] [--index path] [--encoder-log f]
//
// A thin front end over the same SearchEngine the CLI uses; it adds no retrieval logic.
// Endpoints:
//   GET /                     the static page (web/index.html)
//   GET /api/info             collection and index parameters
//   GET /api/search?q=&k=&ef= both backends' top-k with scores, latencies, recall, speedup
//   GET /img/<internal id>    an indexed image (served only by numeric id: no path input)
// Binds to 127.0.0.1 by default so the demo is not exposed to the network. Searches are
// serialised with a mutex: HNSWIndex::search and the encoder helper are single-threaded
// by design.
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>

#include <csignal>
#include <pthread.h>

#include <httplib.h>

#include "app_common.hpp"
#include "compare.hpp"
#include "data/errors.hpp"
#include "embedding/text_encoder_process.hpp"
#include "index/brute_force_index.hpp"
#include "persistence/index_manager.hpp"

using namespace sir;
using nlohmann::json;

namespace {

constexpr int kMaxK = 50;       // keeps the page readable; the engine itself allows K <= N
constexpr int kMaxEf = 1000;

std::string readFile(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw DataError("cannot read " + p.string());
    std::ostringstream os;
    os << in.rdbuf();
    return os.str();
}

int intParam(const httplib::Request& req, const char* name, int def) {
    if (!req.has_param(name)) return def;
    const std::string v = req.get_param_value(name);
    std::size_t used = 0;
    int out = 0;
    try {
        out = std::stoi(v, &used);
    } catch (const std::exception&) {
        throw QueryError(std::string(name) + " must be an integer, got '" + v + "'");
    }
    if (used != v.size())
        throw QueryError(std::string(name) + " must be an integer, got '" + v + "'");
    return out;
}

json resultsJson(const SearchResponse& r, const MetadataManager& meta,
                 const std::set<int>& other) {
    json arr = json::array();
    for (const SearchResult& s : r.results) {
        const ImageRecord& rec = meta.at(s.internalId);
        arr.push_back({{"rank", s.rank},
                       {"id", s.internalId},
                       {"image_id", s.imageId},
                       {"score", s.score},
                       {"caption", rec.captions.empty() ? "" : rec.captions.front()},
                       {"in_other", other.count(s.internalId) > 0},
                       {"url", "/img/" + std::to_string(s.internalId)}});
    }
    return arr;
}

void sendJson(httplib::Response& res, int status, const json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = app::parseArgs(argc, argv);
        const AppConfig cfg = app::loadAppConfig(args);
        const std::string host = args.get("host", "127.0.0.1");
        const int port = args.getInt("port", 8080);
        const std::string page = readFile(cfg.root / "web" / "index.html");

        std::cerr << "loading metadata, embeddings and index...\n";
        const MetadataManager meta = MetadataManager::load(cfg.metadata, cfg.root);
        const EmbeddingStore store = app::loadImageStore(cfg);
        BruteForceIndex bf(store);
        bf.insertAll();
        HNSWIndex hnsw =
            IndexManager::load(args.get("index", cfg.index.string()), store, cfg.modelName);
        const int defaultEf = cfg.hnsw.efSearch;
        std::cerr << "starting text encoder...\n";
        TextEncoderProcess encoder(cfg.python, cfg.textEncoderServer, cfg.configFile, cfg.dim,
                                   args.get("encoder-log", "/dev/null"));
        const SearchEngine bfEngine(meta, store, bf, &encoder);
        const SearchEngine hnswEngine(meta, store, hnsw, &encoder);
        std::mutex searchMutex;

        httplib::Server svr;
        svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
            res.set_content(page, "text/html; charset=utf-8");
        });
        svr.Get("/api/info", [&](const httplib::Request&, httplib::Response& res) {
            const HnswParams& p = hnsw.params();
            sendJson(res, 200,
                     {{"images", store.size()}, {"dim", store.dim()}, {"model", cfg.modelName},
                      {"M", p.M}, {"ef_construction", p.efConstruction},
                      {"ef_search_default", defaultEf}, {"max_k", kMaxK}, {"max_ef", kMaxEf},
                      {"selection", p.selectHeuristic ? (p.keepPrunedConnections
                                                             ? "heuristic+keepPruned"
                                                             : "heuristic")
                                                      : "naive"}});
        });
        svr.Get("/api/search", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                const std::string q = req.has_param("q") ? req.get_param_value("q") : "";
                const int k = intParam(req, "k", 5);
                const int ef = intParam(req, "ef", defaultEf);
                if (k > kMaxK)
                    throw QueryError("K is limited to " + std::to_string(kMaxK) + " in the web UI");
                if (ef < 1 || ef > kMaxEf)
                    throw QueryError("efSearch must be between 1 and " + std::to_string(kMaxEf));
                const std::lock_guard<std::mutex> lock(searchMutex);
                hnsw.setEfSearch(ef);
                const app::Comparison c = app::compareBackends(q, k, bfEngine, hnswEngine);
                sendJson(res, 200,
                         {{"query", c.query}, {"k", k}, {"ef", ef},
                          {"encode_ms", c.encodeMs}, {"recall", c.recall},
                          {"speedup", c.speedup},
                          {"exact", {{"ms", c.exactMs},
                                     {"results", resultsJson(c.exact, meta, c.approxIds)}}},
                          {"hnsw", {{"ms", c.approxMs},
                                    {"results", resultsJson(c.approx, meta, c.exactIds)}}}});
            } catch (const QueryError& e) {
                sendJson(res, 400, {{"error", e.what()}});
            } catch (const std::exception& e) {
                sendJson(res, 500, {{"error", e.what()}});
            }
        });
        svr.Get(R"(/img/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
            try {
                const int id = std::stoi(req.matches[1].str());
                if (id < 0 || static_cast<std::size_t>(id) >= meta.size()) {
                    res.status = 404;
                    return;
                }
                res.set_content(readFile(meta.at(id).path), "image/jpeg");
                res.set_header("Cache-Control", "max-age=86400");
            } catch (const std::exception&) {
                res.status = 404;
            }
        });

        // Clean shutdown on Ctrl+C / SIGTERM: block the signals in every thread (the
        // server's worker threads inherit the mask) and wait for them in one thread, which
        // stops the server so main returns and all RAII destructors run, including the
        // text-encoder helper's.
        sigset_t stopSignals;
        sigemptyset(&stopSignals);
        sigaddset(&stopSignals, SIGINT);
        sigaddset(&stopSignals, SIGTERM);
        pthread_sigmask(SIG_BLOCK, &stopSignals, nullptr);
        std::thread stopper([&] {
            int sig = 0;
            sigwait(&stopSignals, &sig);
            std::cerr << "\nshutting down...\n";
            svr.stop();
        });
        stopper.detach();

        if (host != "127.0.0.1" && host != "localhost")
            std::cerr << "warning: binding to " << host
                      << " exposes the demo beyond this machine\n";
        std::cerr << "ready: " << store.size() << " images. Open http://" << host << ":" << port
                  << "/  (Ctrl+C to stop)\n";
        if (!svr.listen(host, port)) {
            std::cerr << "error: could not listen on " << host << ":" << port
                      << " (port in use? try --port 8081)\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
