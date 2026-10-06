# GenRec on MovieLens — Plan

Paper: [GenRec: An LLM-Backed Recommendation Ranker at Netflix](https://arxiv.org/html/2608.10257v2)

## Goal
Reproduce GenRec as described in the paper on public data (MovieLens-1M) and check whether it beats a tuned standard baseline (SASRec). Paper-faithful by default; anything not in the paper is opt-in and labeled.

| Component | Status |
|---|---|
| Verbalized history → LLM → pooled `h` → `h·e_i`, CE over full catalog | ✅ faithful |
| Context engineering (prompt versions) | ✅ faithful |
| Reward weighting (here: rating) | ✅ faithful, tested |
| LM loss `β·L_language` on the target title | 🔄 faithful, testing |
| Model-size scaling | ✅ faithful |
| Warmup, grad clip, LayerNorm on `h`, LoRA | implementation details the paper doesn't specify |
| Phase 1 (domain pre-training on Netflix logs) | ❌ not reproducible (no data); Qwen2.5 used as-is |
| Multi-position loss, item init from text | ❌ not in paper — out of scope |

## Idea
```
history → text prompt → LLM → h (last token) → score_i = h · e_i → softmax over all movies
```

## Data
- MovieLens-1M (auto-download from GroupLens): `ratings.dat`, `movies.dat`
- Sort by time per user, keep users with ≥5 ratings
- Split: leave-one-out (last = test, 2nd last = valid)
- Prompt (last 20 movies), `--prompt`:
  ```
  v2: Watched (rating 1-5): Toy Story (1995) 5; Heat (1995) 3; ...\nNext:          (~270 tokens)
  v3: Watched (rating 1-5): Toy Story (1995) [Animation|Children's|Comedy] 5; ...  (~420 tokens)
  v4: "User: male, age 25-34, programmer\n" + v3  (~435 tokens)
  v5: v3 + first synopsis sentence (TMDb, via Kaggle "The Movies Dataset") for the last 5 movies  (~550 tokens)
  ```

## Model
- Backbone: `Qwen/Qwen2.5-{0.5B,1.5B,3B}` + LoRA (r=16, q/k/v/o); 3B is Qwen Research License (non-commercial), others Apache 2.0
- `h` = LayerNorm(hidden state of last token)  — raw `h` norm ~300 → huge logits
- Head: item embeddings `E ∈ R^{N×d}`, `score = h @ E.T`
- Training: 5–10% linear warmup → linear decay, grad clip 1.0 — without it, training can collapse to popularity

## Loss
```
L = (1−β)·CE(score, target_item) + β·LM(target_title)     # paper: α+β=1
```
- `--lm_weight β` (default 0 = ranking only); β=0.1 under test
- Optional: weight loss by rating (paper's "reward")

## Inference
- One forward pass → scores for all movies → mask seen → top-K

## Eval
- MRR, HR@10, NDCG@10 (full catalog)
- Baselines: Popularity, SASRec

## Environment
| | Apple Silicon (MPS) | 16GB GPU (CUDA) | Cloud 24–80GB GPU |
|---|---|---|---|
| Use | Debug, small subset | Main training | Model-size scaling |
| Max model | ~0.5B | 0.5B full / 1.5–3B LoRA | 7B+ |

- Switch with `--device {cpu,mps,cuda}`
- Low memory (e.g. 16GB Apple Silicon): add `--grad_ckpt` (~30% slower, much less memory)
- Windows: use WSL2 + CUDA PyTorch

Measured (RTX 4090, bf16, ~97k samples/epoch; varies ~2× by host):
| Backbone | Prompt | Time/epoch |
|---|---|---|
| 0.5B | v2 | ~12–25 min |
| 0.5B | v3 | ~25 min |
| 1.5B | v2 | ~33 min |
| 1.5B | v3 | ~65 min (needs `--grad_ckpt` on 24GB) |

## Files
```
common.py  # data + split, metrics, Popularity, device
sasrec.py  # SASRec: model + train
genrec.py  # GenRec: model + train
plot.py    # runs/ → docs/figures/*.png
docs/      # plan.md, results.md
runs/      # (gitignored) <model>_<time>/log.txt, metrics.json, history.csv
```
Shared interface: `score_fn(users, histories) -> (B, n_items+1)`
Run: `uv run python {common,sasrec,genrec}.py`

## Steps
| # | Step | Status |
|---|---|---|
| 1 | Data: load, split, verbalize | ✅ |
| 2 | Baselines: Popularity, SASRec | ✅ SASRec tuned NDCG@10 0.182 (untuned 0.168) |
| 3 | Pipeline on small subset (CPU/MPS) | ✅ |
| 4 | First GPU run: 0.5B + LoRA | ✅ beats Popularity |
| 5 | Ablations | 🔄 prompt v2→v3 ✅ (+4%); rating weights ✅ (no gain); v4/v5 ✅ (v5 no gain; v4 +2% at 1.5B, borderline); LM loss 🔄 (#19); history length ⬜ |
| 6 | Scaling: 0.5B → 1.5B → 3B → 7B | 🔄 0.5B, 1.5B ✅ (+7%); 7B ✅ (0.1569, 2 ep); epochs: 1.5B 3→5 ep +5%, 8 ep 🔄 (#18); 3B queued (#20); final 7B (#22) |
| 7 | Data scaling on ML-25M (separate benchmark) | ⬜ after current queue |
| 8 | README: results + reproduce | ⬜ |

## Runs
Machines: **Cloud** = RTX 4090 ($0.74/h) or A100 80GB (~$1.4–1.6/h) · **Local** = RTX 4060 Ti 16GB (~2.6× slower than 4090, $0) · **M1** = Apple M1 (MPS)

| # | Run | Status | Machine | Time | Cost | Test NDCG@10 |
|---|---|---|---|---|---|---|
| 1 | SASRec (untuned: d=64, 100 ep) | ✅ | M1 | 16 min | $0 | 0.1681 |
| 2 | 0.5B v2, 1 ep | ✅ | Cloud 4090 | 26 min | $0.45 | 0.0945 |
| 3 | 0.5B v2, 1 ep, + warmup/clip | ✅ | Cloud 4090 | 13 min | $0.25 | 0.1295 |
| 4 | 0.5B v2, 3 ep, lr 2e-4 | ✅ | Cloud 4090 | 38 min | $0.55 | 0.1441 |
| 5 | 1.5B v2, 3 ep, lr 1e-4 | ✅ | Cloud 4090 | 99 min | $1.40 | 0.1486 |
| 6 | 0.5B v2, 3 ep, lr 1e-4 | ✅ | Cloud 4090 | 74 min | $1.00 | 0.1386 |
| 7 | 0.5B v3, 3 ep, lr 2e-4 | ✅ | Cloud 4090 | 78 min | $1.30 | 0.1500 |
| 8 | 1.5B v3, 3 ep, lr 1e-4 | ✅ | Cloud 4090 | 3.4 h | $2.50 (+$0.85 failed pod) | 0.1546 |
| 9 | 7B v3, 2 ep, lr 5e-5 | ✅ | Cloud A100 80GB | 6.1 h | $9.60 (+$0.65 crashed start) | 0.1569 |
| 10 | 0.5B v3, seed 43 (noise check) | ✅ | Local | 3.4 h | $0 | 0.1558 |
| 11 | 0.5B v3, rating-weighted loss | ✅ no gain | Local | 4.0 h | $0 | 0.1473 |
| 12 | 1.5B v3, 5 ep | ✅ best | Local | 19 h | $0 | **0.1627** |
| 13 | 0.5B v5 | ✅ | Local | 5.6 h | $0 | 0.1520 |
| 14 | 0.5B v4 | ✅ | Local | 4.0 h | $0 | 0.1530 |
| 15 | 1.5B v4 (`--grad_ckpt`) | ✅ | Local | 11.8 h | $0 | 0.1581 |
| 16 | 1.5B v5 (`--grad_ckpt`) | ✅ no gain | Local | 16.8 h | $0 | 0.1552 |
| 17 | SASRec tuning: d × dropout × heads (5 runs, 200 ep) | ✅ new baseline | M1 | 4 h | $0 | **0.1820** |
| 18 | 1.5B v4, 8 ep (epoch-count study for final) | 🔄 running (valid ep1/2: 0.1145 / 0.1388) | Local | ~31 h | $0 | – |
| 19 | 0.5B v4 + LM loss (β=0.1) | ⏳ queued | Local | ~4.5 h | $0 | – |
| 20 | 3B v4 (`--grad_ckpt`) | ⏳ queued | Local | ~17–22 h | $0 | – |
| 21 | 0.5B v3, history 50 (`--grad_ckpt`) | ⏳ queued | Local | ~11 h | $0 | – |
| 22 | **Final: 7B v4, epochs from #18, LM loss per #19, lr 5e-5** | ⬜ after #18, #19 | Cloud A100 80GB | ~9–16 h | ~$15–26 | – |

### SASRec tuning (#17; Mac M1, 200 ep, selected by valid)
| d | dropout | heads | Valid NDCG@10 | Test NDCG@10 |
|---|---|---|---|---|
| 64 | 0.2 | 1 | 0.1772 | 0.1709 |
| 64 | 0.5 | 1 | 0.1452 | 0.1394 |
| 128 | 0.2 | 1 | 0.1938 | 0.1828 |
| 128 | 0.5 | 1 | 0.1605 | 0.1546 |
| **128** | **0.2** | **2** | **0.1942** | **0.1820** ← baseline |

### Final model goal
Tuned SASRec (0.1820) is the bar; best GenRec so far is 0.1627 (−11%). Projection for 7B v4 at 3–5 ep: ~0.165–0.172 (−5% to −9%) — likely still below SASRec. Goal: close as much of the gap as possible, report the remaining gap honestly.

### Prompt × size grid (test NDCG@10)
Settings fixed per size (0.5B: lr 2e-4, 5% warmup; 1.5B/3B: lr 1e-4, 10% warmup; 7B: lr 5e-5). 7B is reserved for the final model.

| Size \ prompt | v2 | v3 | v4 | v5 |
|---|---|---|---|---|
| 0.5B | 0.1441 (#4) | 0.1500 (#7), seed 43: 0.1558 (#10) | 0.1530 (#14) | 0.1520 (#13) |
| 1.5B | 0.1486 (#5) | 0.1546 (#8), 5 ep: 0.1627 (#12) | 0.1581 (#15) | 0.1552 (#16) |
| 3B | – | – | #20 | – |
| 7B | – | 0.1569 (#9, 2 ep) | final #22 | – |

### Planned: ML-25M data scaling (step 7)
Fixed model (0.5B v3), vary #users sampled from ML-25M; SASRec at each size. Eval on a fixed random 10k-user subset; full catalog (~59k movies). Not comparable to ML-1M numbers.

| # | Run | Status | Machine | Time | Cost | Test NDCG@10 |
|---|---|---|---|---|---|---|
| 23 | Code: ML-25M loader, `--dataset`, user subsample, eval subset | ⬜ | – | – | – | – |
| 24 | SASRec, 10k / 40k / 160k users | ⬜ | Local | ~1–3 h total | $0 | – |
| 25 | GenRec 0.5B v3, 10k users | ⬜ | Local | ~5 h (3 ep) | $0 | – |
| 26 | GenRec 0.5B v3, 40k users | ⬜ | Local | ~21 h (3 ep) | $0 | – |
| 27 | GenRec 0.5B v3, 160k users (optional) | ⬜ | Local / Cloud | ~28 h/epoch Local | $0 / ~$8/epoch | – |

Results → [results.md](results.md)
