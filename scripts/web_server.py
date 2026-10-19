#!/usr/bin/env python3
"""Local web page for the search engine.

    .venv/bin/python scripts/web_server.py [--port 8080] [--host 127.0.0.1]
                                           [--index path] [--encoder-log file]

Python only serves the page and the image files. Every search is done by the C++ engine
(build/sir_cli --serve), which this script starts once and talks to through its
stdin/stdout: one request line "k<TAB>ef<TAB>query" in, one JSON line out.

Endpoints:
  GET /                        web/index.html
  GET /api/info                collection and index parameters
  GET /api/search?q=&k=&ef=    both indexes' top-k with scores, latencies, recall, speedup
  GET /img/<internal id>       an indexed image, by numeric id only (no path from the client)

Binds to 127.0.0.1 by default, so the demo is not exposed to the network. The engine
answers one request at a time, so searches are serialised with a lock.
"""
import argparse
import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.config import ROOT, load_config  # noqa: E402

IMG_PATH = re.compile(r"^/img/(\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--config", default=str(ROOT / "config" / "config.json"))
    ap.add_argument("--index", help="index file (default: paths.index from the config)")
    ap.add_argument("--encoder-log", default="/dev/null")
    args = ap.parse_args()

    cfg = load_config(args.config)
    page = (ROOT / "web" / "index.html").read_bytes()
    meta = json.loads((ROOT / cfg["paths"]["metadata"]).read_text(encoding="utf-8"))
    image_paths = [ROOT / r["path"] for r in meta["records"]]  # index = internal id

    cmd = [str(ROOT / "build" / "sir_cli"), "--serve", "--config", args.config,
           "--encoder-log", args.encoder_log]
    if args.index:
        cmd += ["--index", args.index]
    engine = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, encoding="utf-8", bufsize=1)
    info_line = engine.stdout.readline()  # engine prints its info once it is ready
    if not info_line:
        sys.exit("error: the C++ engine (build/sir_cli --serve) failed to start; see above")
    info = json.loads(info_line)
    engine_lock = threading.Lock()

    def search(q: str, k: int, ef: int) -> tuple[int, str]:
        q = q.replace("\t", " ").replace("\r", " ").replace("\n", " ")  # one line per request
        with engine_lock:
            engine.stdin.write(f"{k}\t{ef}\t{q}\n")
            engine.stdin.flush()
            line = engine.stdout.readline()
        if not line:
            return 500, json.dumps({"error": "the search engine process has stopped"})
        return json.loads(line).get("status", 500), line

    class Handler(BaseHTTPRequestHandler):
        def send(self, status: int, body: bytes, content_type: str, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, text: str) -> None:
            self.send(status, text.encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802 (name fixed by http.server)
            url = urlsplit(self.path)
            if url.path == "/":
                self.send(200, page, "text/html; charset=utf-8")
            elif url.path == "/api/info":
                self.send_json(200, info_line)
            elif url.path == "/api/search":
                params = parse_qs(url.query)
                q = params.get("q", [""])[0]
                try:
                    k = int(params.get("k", ["5"])[0])
                    ef = int(params.get("ef", [str(info["ef_search_default"])])[0])
                except ValueError:
                    self.send_json(400, json.dumps({"error": "k and ef must be integers"}))
                    return
                status, body = search(q, k, ef)
                self.send_json(status, body)
            elif (m := IMG_PATH.match(url.path)) and int(m.group(1)) < len(image_paths):
                try:
                    data = image_paths[int(m.group(1))].read_bytes()
                except OSError:
                    self.send(404, b"not found", "text/plain")
                    return
                self.send(200, data, "image/jpeg", cache=True)
            else:
                self.send(404, b"not found", "text/plain")

        def log_message(self, fmt: str, *a) -> None:  # keep the console quiet
            pass

    if args.host not in ("127.0.0.1", "localhost"):
        print(f"warning: binding to {args.host} exposes the demo beyond this machine",
              file=sys.stderr)
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        engine.stdin.close()
        engine.wait()
        sys.exit(f"error: could not listen on {args.host}:{args.port} ({e}); try --port 8081")
    print(f"ready: {info['images']} images. Open http://{args.host}:{args.port}/  (Ctrl+C to stop)",
          file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...", file=sys.stderr)
    finally:
        server.server_close()
        engine.stdin.close()  # end of input: the engine exits and stops its text encoder
        engine.wait(timeout=15)


if __name__ == "__main__":
    main()
