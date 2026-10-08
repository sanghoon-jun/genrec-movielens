# GenRec on MovieLens — Plan

Paper: [GenRec: An LLM-Backed Recommendation Ranker at Netflix](https://arxiv.org/html/2608.10257v2)

## Goal
Reproduce GenRec as described in the paper on public data (MovieLens-1M) and check whether it beats a tuned standard baseline (SASRec). Paper-faithful by default; anything not in the paper is opt-in and labeled.

| Component | Status |
|---|---|
| Verbalized history → LLM → pooled `h` → `h·e_i`, CE over full catalog | ✅ faithful |
| Context engineering (prompt versions, history length) | ✅ faithful |
| LM loss `β·L_language` on the target title | ✅ faithful (+2.9% at 0.5B, within noise; kept) |
| Reward weighting (here: rating) | ✅ faithful, tested (no gain; not used) |
| Model-size scaling | ✅ faithful |
| Warmup, grad clip, LayerNorm on `h`, LoRA | implementation details the paper doesn't specify |
| Phase 1 (domain pre-training on Netflix logs) | ❌ not reproducible (no data); Qwen2.5 used as-is |

## Faithful recipe (final)
```
prompt v4 (user profile + last 20 movies: title [genres] rating) → Qwen2.5 + LoRA
→ h = LayerNorm(last-token hidden state) → score = h · E → CE over all movies
L = 0.9·L_rank + 0.1·L_lm(target title), 3 epochs, warmup → linear decay, grad clip 1.0
```
| Size | LR (LoRA) | Warmup | Notes |
|---|---|---|---|
| 0.5B | 2e-4 | 5% | |
| 1.5B / 3B | 1e-4 | 10% | `--grad_ckpt` on 16GB |
| 7B | 5e-5 | 10% | A100 80GB, `--grad_ckpt` |

## Data
- MovieLens-1M (auto-download from GroupLens); users with ≥5 ratings, sorted by time
- Split: leave-one-out (last = test, 2nd last = valid)
- Prompt versions (`--prompt`):
  ```
  v2: Watched (rating 1-5): Toy Story (1995) 5; Heat (1995) 3; ...\nNext:          (~270 tokens)
  v3: Watched (rating 1-5): Toy Story (1995) [Animation|Children's|Comedy] 5; ...  (~420 tokens)
  v4: "User: male, age 25-34, programmer\n" + v3  (~435 tokens)
  v5: v3 + first synopsis sentence (TMDb, via Kaggle "The Movies Dataset") for the last 5 movies  (~550 tokens)
  ```

## Eval
- MRR, HR@10, NDCG@10 over the full catalog (seen items masked)
- Baselines: Popularity, SASRec (tuned)
- Run selection by valid NDCG@10 only; test reported once

## Environment
- `--device {cpu,mps,cuda}`; `--grad_ckpt` for low memory (~30% slower)
- Machines: **Cloud** = RTX 4090 ($0.74/h) or A100 80GB (~$1.6/h) · **Local** = RTX 4060 Ti 16GB ($0) · **M1** = Apple M1 (MPS)
- 3B is Qwen Research License (non-commercial); 0.5B / 1.5B / 7B are Apache 2.0

## Files
```
common.py  # data + split, metrics, Popularity, run logging
sasrec.py  # SASRec: model + train
genrec.py  # GenRec: model + train (--prompt, --lm_weight, --rating_weight, --resume)
plot.py    # runs/ → docs/figures/*.png
docs/      # plan.md, results.md
runs/      # (gitignored) <model>_<time>/log.txt, metrics.json, history.csv, *.pt
```

## Steps
| # | Step | Status |
|---|---|---|
| 1 | Data, split, verbalize | ✅ |
| 2 | Baselines: Popularity, SASRec (tuned) | ✅ SASRec 0.1820 |
| 3 | GenRec pipeline + training stability (warmup, clip) | ✅ |
| 4 | Development ablations: prompts, LM loss, rating weights, epochs | ✅ (history length ⏳ #29) |
| 5 | Full-recipe grid (★): v3/v4/v5 × 0.5B/1.5B/3B + 7B v4 | 🔄 0.5B v4 ✅; #20–28 running/queued (~5 days Local) |
| 6 | README: results + reproduce | ⬜ |

## Runs
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
| 12 | 1.5B v3, 5 ep | ✅ | Local | 19 h | $0 | 0.1627 |
| 13 | 0.5B v5 | ✅ | Local | 5.6 h | $0 | 0.1520 |
| 14 | 0.5B v4 | ✅ | Local | 4.0 h | $0 | 0.1530 |
| 15 | 1.5B v4 | ✅ | Local | 11.8 h | $0 | 0.1581 |
| 16 | 1.5B v5 | ✅ no gain | Local | 16.8 h | $0 | 0.1552 |
| 17 | SASRec tuning: d × dropout × heads (5 runs, 200 ep) | ✅ baseline | M1 | 4 h | $0 | **0.1820** |
| 18 | 1.5B v4, 8 ep | ✅ no gain vs 3 ep | Local | 31.8 h | $0 | 0.1558 |
| 19 | ★ 0.5B v4 + LM loss | ✅ (+2.9% vs #14) | Local | 4.3 h | $0 | 0.1575 |
| 20 | ★ 3B v4 + LM loss | 🔄 eval | Local | ~25 h | $0 | – |
| 21 | ★ 1.5B v4 + LM loss | ⏳ queued | Local | ~12.5 h | $0 | – |
| 22 | ★ **Final: 7B v4 + LM loss** | ⬜ ready | Cloud A100 80GB | ~9–10 h | ~$15–16 | – |
| 23 | ★ 0.5B v3 + LM loss | ⏳ queued | Local | ~4 h | $0 | – |
| 24 | ★ 0.5B v5 + LM loss | ⏳ queued | Local | ~6 h | $0 | – |
| 25 | ★ 1.5B v3 + LM loss | ⏳ queued | Local | ~12 h | $0 | – |
| 26 | ★ 1.5B v5 + LM loss | ⏳ queued | Local | ~17 h | $0 | – |
| 27 | ★ 3B v3 + LM loss | ⏳ queued | Local | ~24 h | $0 | – |
| 28 | ★ 3B v5 + LM loss | ⏳ queued | Local | ~35 h | $0 | – |
| 29 | 0.5B v4 + LM loss, history 50 (ablation vs #19) | ⏳ queued | Local | ~11 h | $0 | – |

★ = full faithful recipe (LM loss, 3 epochs; prompt v3/v4/v5) → reported in results.md. All other GenRec runs are development runs/ablations, kept here only.

### SASRec tuning (#17; M1, 200 ep, selected by valid)
| d | dropout | heads | Valid NDCG@10 | Test NDCG@10 |
|---|---|---|---|---|
| 64 | 0.2 | 1 | 0.1772 | 0.1709 |
| 64 | 0.5 | 1 | 0.1452 | 0.1394 |
| 128 | 0.2 | 1 | 0.1938 | 0.1828 |
| 128 | 0.5 | 1 | 0.1605 | 0.1546 |
| **128** | **0.2** | **2** | **0.1942** | **0.1820** ← baseline |

### Full-recipe grid ★ (test NDCG@10) — reported in results.md
| Size \ prompt | v3 | v4 | v5 |
|---|---|---|---|
| 0.5B | #23 | 0.1575 (#19) | #24 |
| 1.5B | #25 | #21 | #26 |
| 3B | #27 | #20 | #28 |
| 7B | – | #22 | – |

### Development grid (ranking-only, no LM loss; test NDCG@10, 3 ep)
| Size \ prompt | v2 | v3 | v4 | v5 |
|---|---|---|---|---|
| 0.5B | 0.1441 (#4) | 0.1500 (#7), seed 43: 0.1558 (#10) | 0.1530 (#14) | 0.1520 (#13) |
| 1.5B | 0.1486 (#5) | 0.1546 (#8) | 0.1581 (#15) | 0.1552 (#16) |
| 7B | – | 0.1569 (#9, 2 ep) | – | – |

### Development findings (not in results.md)
- **Stability:** warmup + grad clip required (without them training collapsed to popularity, #3 vs #2); LoRA LR must drop with size (1.5B at 2e-4 stalled)
- **Noise:** two seeds of 0.5B v3 → 0.1500 / 0.1558 (±2%); differences under ~4% are not significant
- **Prompts (ranking-only):** v2 → v3 +4% at both sizes; v4 0 / +2%; v5 no gain at +30–60% compute
- **Size (ranking-only):** 0.5B → 1.5B → 7B flat on test (0.153–0.158)
- **LM loss:** 0.5B v4 0.1530 → 0.1575 (+2.9%, within noise) → adopted as part of the faithful recipe
- **Reward (rating) weighting:** 0.1473, no gain (metric counts the next rating regardless of value) → not used
- **Epochs:** 3 is enough; 5 ep test 0.1627 but valid flat; 8 ep valid 0.1700 → 0.1661 (memorization) → 3 epochs

## Later: beyond the faithful recipe (to discuss)
Not in the paper; only after the faithful results are final, and always labeled.
- Multi-position loss (grade every movie in a prompt; ~16–20× more training signal per compute)
- Item embeddings initialized from text (title / genres / synopsis)
- ML-25M (more items, long tail, global time split) as a second benchmark

Results → [results.md](results.md)
