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
  ```

## Model
- Backbone: `Qwen/Qwen2.5-{0.5B,1.5B}` + LoRA (r=16, q/k/v/o)
- `h` = LayerNorm(hidden state of last token)  — raw `h` norm ~300 → huge logits
- Head: item embeddings `E ∈ R^{N×d}`, `score = h @ E.T`
- Training: 5–10% linear warmup → linear decay, grad clip 1.0 — without it, training can collapse to popularity

## Loss
```
L = α·CE(score, target_item) + β·LM(target_title)
```
- Current: α=1, β=0 (LM loss not implemented yet)
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
| 5 | Ablations | 🔄 prompt v2→v3 ✅ (+4%); LM loss, history length, rating weights ⬜ |
| 6 | Scaling: 0.5B → 1.5B → 7B | 🔄 0.5B, 1.5B ✅ (+7%); 1.5B v3 running; 7B ⬜ |
| 7 | README: results + reproduce | ⬜ |

Results → [results.md](results.md)
