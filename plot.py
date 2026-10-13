"""runs/ -> docs/figures/{curves,scaling}.png

Uses the best main run (full data, seed 42, no ablation flags) per model unless --runs is given.
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

from common import RUNS

OUT = Path(__file__).parent / "docs" / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical order
LINE_COLOR = {"genres": 1, "profile": 2, "synopsis": 3}  # color follows the prompt, never its rank
OLD_PROMPT = {"v2": "rating", "v3": "genres", "v4": "profile", "v5": "synopsis"}  # runs made before the rename
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


def prompt_of(args):
    p = args.get("prompt", "rating")
    return OLD_PROMPT.get(p, p)


def label(path, info):
    model = path.name.split("_")[0]
    if model == "genrec" and info:
        prompt = prompt_of(info["args"])
        return f"GenRec {info['args']['backbone'].split('/')[-1]} {prompt}"
    return {"sasrec": "SASRec"}.get(model, model)


def main_runs():
    """Main runs only: full data, default seed, no ablation flags (no cherry-picking seeds)."""
    for p in sorted(RUNS.glob("*_*")):
        info, _ = read_run(p)
        a = info["args"] if info else {}
        faithful = p.name.startswith("sasrec") or (a.get("lm_weight", 0) > 0 and a.get("epochs") == 3)
        if info and faithful and not a.get("users") and a.get("seed", 42) == 42 and not a.get("rating_weight") \
                and a.get("history", 20) == 20:
            yield p, info  # results.md shows the full faithful recipe only (LM loss, 3 epochs)


def valid_score(p, info):
    """Selection uses validation only (max valid NDCG@10); test is never used to choose."""
    _, hist = read_run(p)
    return max((v for _, v in hist.get("valid_NDCG@10", [])), default=info["metrics"]["NDCG@10"])


def full_runs():
    """Best main run per (model, backbone, prompt, epochs), by valid NDCG@10."""
    best = {}
    for p, info in main_runs():
        key, score = label(p, info), valid_score(p, info)
        if key not in best or score > best[key][1]:
            best[key] = (p, score)
    return [p for p, _ in best.values()]


SCALING_EPOCHS = 3  # scaling plot compares sizes at the same training length


def best_per_size(runs):
    """Curves stay readable: best run per backbone (any prompt/epochs) + SASRec, at most len(SERIES) lines."""
    best = {}
    for p in runs:
        info, _ = read_run(p)
        key = info["args"].get("backbone", "sasrec")
        if key not in best or valid_score(p, info) > best[key][1]:
            best[key] = (p, valid_score(p, info))
    return [p for p, _ in sorted(best.values(), key=lambda x: -x[1])][:len(SERIES)]


def plot_curves(runs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for color, p in zip(SERIES, best_per_size(runs)):
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
    # per (prompt, size): prefer a SCALING_EPOCHS run; otherwise the closest, annotated with its epochs
    cells, sasrec = {}, None
    for p, info in main_runs():
        a = info["args"]
        size = re.search(r"(\d+(?:\.\d+)?)B", a.get("backbone", "")) if p.name.startswith("genrec") else None
        if size:
            line = prompt_of(a)
            key, ep = (line, float(size.group(1))), a.get("epochs", 1)
            rank = (-abs(ep - SCALING_EPOCHS), valid_score(p, info))  # closest to SCALING_EPOCHS, then best valid
            if key not in cells or rank > cells[key][0]:
                cells[key] = (rank, ep, info["metrics"]["NDCG@10"])
    for p in runs:
        if p.name.startswith("sasrec"):
            sasrec = read_run(p)[0]["metrics"]["NDCG@10"]
    points = {}  # prompt -> [(size, ndcg, epochs)]
    for (prompt, size), (_, ep, ndcg) in cells.items():
        points.setdefault(prompt, []).append((size, ndcg, ep))
    if not points:
        print("no GenRec full runs yet: skip scaling.png")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = sorted({x for pts in points.values() for x, _, _ in pts})
    for prompt, pts in sorted(points.items()):
        color = SERIES[LINE_COLOR.get(prompt, 0)]
        xs, ys, eps = zip(*sorted(pts))
        ax.plot(xs, ys, "-o", color=color, markeredgecolor=SURFACE, markeredgewidth=2, label=f"GenRec {prompt}")
        for x, y, ep in zip(xs, ys, eps):  # exact values live in results.md; only flag off-schedule points
            if ep != SCALING_EPOCHS:
                ax.annotate(f"{ep} ep", (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    for y, name, style in ((sasrec, "SASRec", "--"),):  # Popularity (~0.02) is far below the zoomed axis
        if y is not None:
            ax.axhline(y, color=INK2, linestyle=style, linewidth=1.2, label=name)
            ax.annotate(name, (1, y), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                        textcoords="offset points", ha="right", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{x:g}B" for x in sizes])
    ax.minorticks_off()
    ax.set(title="Test NDCG@10 vs backbone size (full faithful recipe)", xlabel="backbone parameters", ylabel="NDCG@10", ylim=(0.13, None))
    ax.title.set_color(INK)
    ax.legend(loc="lower right", fontsize=9)
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
