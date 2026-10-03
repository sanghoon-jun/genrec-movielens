"""Shared: MovieLens-1M data + split, full-catalog metrics, Popularity, device.

Any model = score_fn(users, histories) -> (B, n_items + 1) scores.
"""
import csv
import io
import json
import math
import platform
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ROOT = Path(__file__).parent / "data"
RUNS = Path(__file__).parent / "runs"


@dataclass
class Data:
    n_items: int             # item ids are 1..n_items, 0 = padding
    seqs: list[list[int]]    # per user, item ids sorted by time
    ratings: list[list[int]] # per user, aligned with seqs
    movies: pd.DataFrame     # index = item id, columns: title, genres

    def split(self, name):
        """Returns (histories, targets). train: history only, no target."""
        if name == "train":
            return [s[:-2] for s in self.seqs], None
        if name == "valid":
            return [s[:-2] for s in self.seqs], [s[-2] for s in self.seqs]
        if name == "test":
            return [s[:-1] for s in self.seqs], [s[-1] for s in self.seqs]
        raise ValueError(name)


def download():
    if not (ROOT / "ml-1m").exists():
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(ROOT)
    return ROOT / "ml-1m"


def load(min_len=5):
    path = download()
    read = lambda f, cols: pd.read_csv(path / f, sep="::", engine="python", names=cols, encoding="latin-1")
    ratings = read("ratings.dat", ["user", "movie", "rating", "ts"])
    movies = read("movies.dat", ["movie", "title", "genres"])

    ratings = ratings[ratings.groupby("user").movie.transform("size") >= min_len]
    movie_ids = sorted(ratings.movie.unique())
    to_item = {m: i + 1 for i, m in enumerate(movie_ids)}
    ratings = ratings.assign(item=ratings.movie.map(to_item)).sort_values(["user", "ts"], kind="stable")

    groups = ratings.groupby("user")
    movies = movies[movies.movie.isin(to_item)].assign(item=lambda d: d.movie.map(to_item)).set_index("item").sort_index()
    return Data(
        n_items=len(movie_ids),
        seqs=groups.item.apply(list).tolist(),
        ratings=groups.rating.apply(list).tolist(),
        movies=movies[["title", "genres"]],
    )



@torch.no_grad()
def evaluate(score_fn, d, split="test", k=10, batch_size=256):
    histories, targets = d.split(split)
    mrr = hr = ndcg = 0.0
    for i in range(0, len(targets), batch_size):
        hist, tgt = histories[i:i + batch_size], targets[i:i + batch_size]
        scores = score_fn(range(i, i + len(hist)), hist).float().cpu()
        scores[:, 0] = -math.inf                     # padding
        for row, h in enumerate(hist):
            scores[row, h] = -math.inf               # already seen
        tgt = torch.tensor(tgt)
        target_score = scores[torch.arange(len(tgt)), tgt]
        rank = (scores > target_score[:, None]).sum(1) + 1  # 1-based
        mrr += (1.0 / rank).sum().item()
        hr += (rank <= k).sum().item()
        ndcg += ((rank <= k) / torch.log2(rank + 1.0)).sum().item()
    n = len(targets)
    return {"MRR": mrr / n, f"HR@{k}": hr / n, f"NDCG@{k}": ndcg / n}


def popularity(d):
    counts = Counter(i for s in d.split("train")[0] for i in s)
    pop = torch.zeros(d.n_items + 1)
    for i, c in counts.items():
        pop[i] = c
    return lambda users, hist: pop.expand(len(hist), -1).clone()


def pick_device(name=None):
    return name or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def machine(device):
    if device == "cuda":
        return torch.cuda.get_device_name()
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return platform.processor() or "cpu"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            s.write(x)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def __getattr__(self, name):  # isatty, fileno, encoding, ... -> real stdout
        return getattr(self.streams[0], name)


def start_run(name, args):
    """runs/<name>_<time>/: log.txt gets everything printed."""
    run = RUNS / f"{name}_{datetime.now():%Y-%m-%d_%H%M%S}"
    run.mkdir(parents=True)
    sys.stdout = Tee(sys.__stdout__, open(run / "log.txt", "w"))
    print(f"run dir: {run}\nargs: {vars(args)}")
    return run


def log_history(run, step, **values):
    """Append rows (step, key, value) to runs/<run>/history.csv."""
    path = run / "history.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step", "key", "value"])
        for k, v in values.items():
            w.writerow([step, k, f"{v:.6g}"])


def save_metrics(run, args, metrics, minutes, device):
    info = {"metrics": metrics, "train_min": round(minutes, 1), "machine": machine(device),
            "device": device, "args": vars(args)}
    (run / "metrics.json").write_text(json.dumps(info, indent=2))


def fmt(m):
    return "  ".join(f"{k}={v:.4f}" for k, v in m.items())


if __name__ == "__main__":
    d = load()
    lens = [len(s) for s in d.seqs]
    print(f"users={len(d.seqs)} items={d.n_items} interactions={sum(lens)} median_len={sorted(lens)[len(lens) // 2]}")
    print("Popularity  ", fmt(evaluate(popularity(d), d)))
