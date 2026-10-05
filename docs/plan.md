# GenRec on MovieLens — Plan

Paper: [GenRec: An LLM-Backed Recommendation Ranker at Netflix](https://arxiv.org/html/2608.10257v2)

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
L = CE(score, target_item)
```
- Paper's LM loss (β·LM(target_title)) dropped: redundant with ranking CE here; paper uses it to keep general language ability
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
| 2 | Baselines: Popularity, SASRec | ✅ SASRec NDCG@10 0.168 |
| 3 | Pipeline on small subset (CPU/MPS) | ✅ |
| 4 | First GPU run: 0.5B + LoRA | ✅ beats Popularity |
| 5 | Ablations | 🔄 prompt v2→v3 ✅ (+4%); rating weights ✅ (no gain); v4/v5 ✅ at 0.5B (within noise); history length ⬜ |
| 6 | Scaling: 0.5B → 1.5B → 3B → 7B | 🔄 0.5B, 1.5B ✅ (+7%); 1.5B v3 ✅ (0.1546); 7B ✅ (0.1569, 2 ep); 3B queued |
| 7 | Data scaling on ML-25M (separate benchmark) | ⬜ after current queue |
| 8 | README: results + reproduce | ⬜ |

## Runs
Machines: **RunPod** = RTX 4090 Secure ($0.74/h) or A100 80GB (~$1.4–1.6/h) · **WSL** = local RTX 4060 Ti 16GB (~2.6× slower than 4090, $0) · **M1** = Apple M1 (MPS)

| # | Run | Status | Machine | Time | Cost | Test NDCG@10 |
|---|---|---|---|---|---|---|
| 1 | SASRec | ✅ | M1 | 16 min | $0 | 0.1681 |
| 2 | 0.5B v2, 1 ep | ✅ | RunPod 4090 | 26 min | $0.45 | 0.0945 |
| 3 | 0.5B v2, 1 ep, + warmup/clip | ✅ | RunPod 4090 | 13 min | $0.25 | 0.1295 |
| 4 | 0.5B v2, 3 ep, lr 2e-4 | ✅ | RunPod 4090 | 38 min | $0.55 | 0.1441 |
| 5 | 1.5B v2, 3 ep, lr 1e-4 | ✅ | RunPod 4090 | 99 min | $1.40 | 0.1486 |
| 6 | 0.5B v2, 3 ep, lr 1e-4 | ✅ | RunPod 4090 | 74 min | $1.00 | 0.1386 |
| 7 | 0.5B v3, 3 ep, lr 2e-4 | ✅ | RunPod 4090 | 78 min | $1.30 | 0.1500 |
| 8 | 1.5B v3, 3 ep, lr 1e-4 | ✅ | RunPod 4090 | 3.4 h | $2.50 (+$0.85 failed pod) | 0.1546 |
| 9 | 7B v3, 2 ep, lr 5e-5 | ✅ | RunPod A100 80GB | 6.1 h | $9.60 (+$0.65 crashed start) | 0.1569 |
| 10 | 0.5B v3, seed 43 (noise check) | ✅ | WSL | 3.4 h | $0 | 0.1558 |
| 11 | 0.5B v3, rating-weighted loss | ✅ no gain | WSL | 4.0 h | $0 | 0.1473 |
| 12 | 1.5B v3, 5 ep | ✅ best | WSL | 19 h | $0 | **0.1627** |
| 13 | 0.5B v5 | ✅ | WSL | 5.6 h | $0 | 0.1520 |
| 14 | 0.5B v4 | ✅ | WSL | 4.0 h | $0 | 0.1530 |
| 15 | 1.5B v4 (`--grad_ckpt`) | ✅ | WSL | 11.8 h | $0 | 0.1581 |
| 16 | 1.5B v5 (`--grad_ckpt`) | 🔄 running | WSL | ~17 h | $0 | – |
| 17 | 1.5B v4, 8 ep (epoch-count study for final) | ⏳ queued | WSL | ~31 h | $0 | – |
| 18 | 3B v4 (`--grad_ckpt`) | ⏳ queued | WSL | ~17–22 h | $0 | – |
| 19 | 0.5B v3, history 50 (`--grad_ckpt`) | ⏳ queued | WSL | ~11 h | $0 | – |
| 20 | **Final: 7B v4, epochs from #17 (3–5), lr 5e-5** | ⬜ after #17 | RunPod A100 80GB | ~9–16 h | ~$15–26 | – |

### Prompt × size grid (test NDCG@10)
Settings fixed per size (0.5B: lr 2e-4, 5% warmup; 1.5B/3B: lr 1e-4, 10% warmup; 7B: lr 5e-5). 7B is reserved for the final model.

| Size \ prompt | v2 | v3 | v4 | v5 |
|---|---|---|---|---|
| 0.5B | 0.1441 (#4) | 0.1500 (#7), seed 43: 0.1558 (#10) | 0.1530 (#14) | 0.1520 (#13) |
| 1.5B | 0.1486 (#5) | 0.1546 (#8), 5 ep: 0.1627 (#12) | 0.1581 (#15) | #16 |
| 3B | – | – | #18 | – |
| 7B | – | 0.1569 (#9, 2 ep) | final #20 | – |

### Planned: ML-25M data scaling (step 7)
Fixed model (0.5B v3), vary #users sampled from ML-25M; SASRec at each size. Eval on a fixed random 10k-user subset; full catalog (~59k movies). Not comparable to ML-1M numbers.

| # | Run | Status | Machine | Time | Cost | Test NDCG@10 |
|---|---|---|---|---|---|---|
| 21 | Code: ML-25M loader, `--dataset`, user subsample, eval subset | ⬜ | – | – | – | – |
| 22 | SASRec, 10k / 40k / 160k users | ⬜ | WSL | ~1–3 h total | $0 | – |
| 23 | GenRec 0.5B v3, 10k users | ⬜ | WSL | ~5 h (3 ep) | $0 | – |
| 24 | GenRec 0.5B v3, 40k users | ⬜ | WSL | ~21 h (3 ep) | $0 | – |
| 25 | GenRec 0.5B v3, 160k users (optional) | ⬜ | WSL / RunPod | ~28 h/epoch WSL | $0 / ~$8/epoch | – |

Results → [results.md](results.md)
