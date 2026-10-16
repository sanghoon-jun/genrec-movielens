# Plan

Remaining work. Overview → [README](../README.md) · results → [results.md](results.md)

## Remaining
| Task | Status | Machine | Time | Cost |
|---|---|---|---|---|
| GenRec-AP 0.5B profile (`--all_positions`) | ✅ test 0.1926 (see results) | Local | 2.7 h | $0 |
| 0.5B profile + LM loss, ratings fixed (rerun of the faithful recipe) | ✅ test 0.1562 (bug version 0.1575: no effect) | Local | 3.9 h | $0 |
| GenRec-AP grid: 0.5B genres, synopsis | 🔄 running (genres), then queued | Local | ~2.5 h + ~4 h | $0 |
| GenRec-AP grid: 1.5B genres, profile, synopsis | ⏳ queued | Local | ~8 + 8 + 11 h | $0 |
| GenRec-AP grid: 3B genres, profile, synopsis | ⏳ queued | Local | ~16 + 16 + 22 h | $0 |
| 0.5B profile + LM loss, history 50 (ablation) | ❌ stopped 2026-10-12 in epoch 2 (epoch 1 checkpoint kept) | Local | ~30 h | $0 |

## Later: beyond the faithful recipe (to discuss)
Not in the paper; only after the faithful results are final, and always labeled.
- ~~Multi-position loss~~ → GenRec-AP (`--all_positions`), now running: the next movie is ranked after every movie in the prompt (~10× the labels per epoch with ~55K prompts instead of ~97K); compare with SASRec (chunked), which gets the same all-position training
- Item embeddings initialized from text (title / genres / synopsis)
- ML-25M (more items, long tail, global time split) as a second benchmark
