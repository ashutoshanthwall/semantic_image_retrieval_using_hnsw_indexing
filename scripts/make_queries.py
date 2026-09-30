#!/usr/bin/env python3
"""Build evaluation/queries.txt: the 5 hand-written G4 queries followed by caption #1
(captions[0]) of `num_heldout_queries` images sampled with a fixed seed. For each held-out
query the source image is the ground-truth positive (recorded in query_ground_truth.csv).
Captions are never used to build embeddings or indexes, so they are held out by design."""
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.config import ROOT, load_config  # noqa: E402

HANDWRITTEN = ["a dog playing in snow", "two men playing guitars on a stage",
               "a child on a red bicycle", "people eating at an outdoor restaurant",
               "a surfer riding a large wave"]
cfg = load_config()
ev = cfg["evaluation"]
meta = json.loads((ROOT / cfg["paths"]["metadata"]).read_text(encoding="utf-8"))
rng = random.Random(ev["seed"])
ids = sorted(rng.sample(range(meta["count"]), ev["num_heldout_queries"]))
ci = ev["heldout_caption_index"]
rows = [("handwritten", "", -1, q) for q in HANDWRITTEN]
for i in ids:
    r = meta["records"][i]
    rows.append(("heldout", r["image_id"], i, r["captions"][ci]))
for *_, q in rows:
    assert q and "\n" not in q and "\t" not in q
(ROOT / cfg["paths"]["queries"]).write_text("".join(q + "\n" for *_, q in rows),
                                            encoding="utf-8")
with open(ROOT / cfg["paths"]["query_ground_truth"], "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["line", "kind", "image_id", "internal_id"])
    for n, (kind, image_id, iid, _) in enumerate(rows):
        w.writerow([n, kind, image_id, iid])
print(f"wrote {len(rows)} queries ({len(HANDWRITTEN)} hand-written, {len(ids)} held-out)")
