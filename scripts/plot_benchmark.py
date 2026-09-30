#!/usr/bin/env python3
"""Recall@10-vs-latency trade-off plot from evaluation/benchmark_results/latency_recall.csv
(measured by sir_benchmark). One axis; series identified by legend, marker shape and
direct labels (not colour alone)."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir.config import ROOT  # noqa: E402

res = ROOT / "evaluation/benchmark_results"
rows = list(csv.DictReader(open(res / "latency_recall.csv")))
SERIES = [("heuristic+keepPruned", "#2a78d6", "o"), ("heuristic", "#eb6834", "s"),
          ("naive", "#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
for name, color, marker in SERIES:
    pts = sorted((float(r["median_ms"]), float(r["recall_at_10"]), int(r["ef_search"]))
                 for r in rows if r["variant"] == name)
    xs, ys, efs = zip(*pts)
    ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8, label=name,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5, zorder=3)
    if name == "heuristic+keepPruned":
        for x, y, ef in pts:
            ax.annotate(f"ef={ef}", (x, y), textcoords="offset points", xytext=(-8, 8),
                        ha="right", color=MUTED, fontsize=8)
bf = next(r for r in rows if r["backend"] == "brute_force")
ax.scatter([float(bf["median_ms"])], [1.0], color=INK, marker="D", s=50, zorder=3,
           label="brute force (exact)")
ax.annotate(f"brute force\n{float(bf['median_ms']):.2f} ms", (float(bf["median_ms"]), 1.0),
            textcoords="offset points", xytext=(-10, -28), ha="right", color=INK, fontsize=8)
ax.set_xscale("log")
ticks = [0.15, 0.2, 0.3, 0.5, 0.75, 1, 2, 3, 5, 8]
ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel("Median retrieval latency per query (ms, log scale, single thread)")
ax.set_ylabel("Recall@10 vs brute force")
ax.set_ylim(0.65, 1.02)
ax.grid(True, which="both", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("HNSW recall/latency trade-off on Flickr30k (31,013 images, 1,005 queries)\n"
             "M=16, efConstruction=200, efSearch ∈ {20, 40, 80, 120}", color=INK, fontsize=10,
             loc="left")
ax.legend(frameon=False, loc="lower right", labelcolor=INK)
fig.tight_layout()
fig.savefig(res / "recall_vs_latency.png", facecolor=fig.get_facecolor())
print("wrote", res / "recall_vs_latency.png")
