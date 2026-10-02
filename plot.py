"""runs/ -> docs/figures/{curves,scaling}.png

Uses the best full-data run (no --users, by test NDCG@10) per model unless --runs is given.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from common import RUNS, evaluate, load, popularity

OUT = Path(__file__).parent / "docs" / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "lines.linewidth": 2, "lines.markersize": 8, "legend.frameon": False,
})


def read_run(path):
    info = json.loads((path / "metrics.json").read_text()) if (path / "metrics.json").exists() else None
    hist = {}
    if (path / "history.csv").exists():
        for row in csv.DictReader(open(path / "history.csv")):
            hist.setdefault(row["key"], []).append((int(row["step"]), float(row["value"])))
    return info, hist


def label(path, info):
    model = path.name.split("_")[0]
    if model == "genrec" and info:
        prompt = info["args"].get("prompt", "v2")
        return f"GenRec {info['args']['backbone'].split('/')[-1]}" + ("" if prompt == "v2" else f" {prompt}")
    return {"sasrec": "SASRec"}.get(model, model)


def full_runs():
    """Best finished full-data run per (model, backbone), by test NDCG@10."""
    best = {}
    for p in sorted(RUNS.glob("*_*")):
        info, _ = read_run(p)
        if info and not info["args"].get("users"):
            key, score = label(p, info), info["metrics"]["NDCG@10"]
            if key not in best or score > best[key][1]:
                best[key] = (p, score)
    return [p for p, _ in best.values()]


def plot_curves(runs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for color, p in zip(SERIES, runs):
        info, hist = read_run(p)
        name = label(p, info)
        for ax, key in ((a1, "loss"), (a2, "valid_NDCG@10")):
            if key in hist:
                xs, ys = zip(*hist[key])
                ax.plot(xs, ys, color=color, label=name)
                ax.plot(xs[-1], ys[-1], "o", color=color, markeredgecolor=SURFACE, markeredgewidth=2)
    a1.set(title="Training loss", xlabel="step", ylabel="cross-entropy")
    a2.set(title="Valid NDCG@10", xlabel="step", ylabel="NDCG@10")
    for ax in (a1, a2):
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0"))
        ax.title.set_color(INK)
    a1.legend(loc="upper right", fontsize=9)
    a2.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=150)
    print(f"wrote {OUT / 'curves.png'}")


def plot_scaling(runs):
    points, sasrec = {}, None  # prompt -> [(size, ndcg)]
    for p in runs:
        info, _ = read_run(p)
        if p.name.startswith("genrec"):
            size = re.search(r"(\d+(?:\.\d+)?)B", info["args"]["backbone"])
            if size:
                points.setdefault(info["args"].get("prompt", "v2"), []).append((float(size.group(1)), info["metrics"]["NDCG@10"]))
        elif p.name.startswith("sasrec"):
            sasrec = info["metrics"]["NDCG@10"]
    if not points:
        print("no GenRec full runs yet: skip scaling.png")
        return
    d = load()
    pop = evaluate(popularity(d), d)["NDCG@10"]

    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = sorted({x for pts in points.values() for x, _ in pts})
    for i, (color, (prompt, pts)) in enumerate(zip(SERIES, sorted(points.items()))):
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, "-o", color=color, markeredgecolor=SURFACE, markeredgewidth=2, label=f"GenRec ({prompt} prompt)")
        offset, ha = ((0, -14), "center") if i == 0 else ((8, 4), "left")  # keep same-size labels apart
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), xytext=offset, textcoords="offset points", ha=ha, fontsize=9, color=INK2)
    for y, name, style in ((sasrec, "SASRec", "--"), (pop, "Popularity", ":")):
        if y is not None:
            ax.axhline(y, color=INK2, linestyle=style, linewidth=1.2, label=name)
            ax.annotate(name, (1, y), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                        textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{x:g}B" for x in sizes])
    ax.minorticks_off()
    ax.set(title="Test NDCG@10 vs backbone size", xlabel="backbone parameters", ylabel="NDCG@10", ylim=(0, None))
    ax.title.set_color(INK)
    ax.legend(loc="center left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=150)
    print(f"wrote {OUT / 'scaling.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run dirs (default: latest full run per model)")
    args = ap.parse_args()
    runs = [Path(r) for r in args.runs] if args.runs else full_runs()
    OUT.mkdir(parents=True, exist_ok=True)
    print("runs:", *[r.name for r in runs])
    plot_curves(runs)
    plot_scaling(runs)
