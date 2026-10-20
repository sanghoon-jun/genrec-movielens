# Plan

Remaining work. Overview → [README](../README.md) · results → [results.md](results.md)

## Remaining
| Task | Status | Machine | Time | Cost |
|---|---|---|---|---|
| GenRec-AP 0.5B profile (`--all_positions`) | ✅ test 0.1926 (see results) | Local | 2.7 h | $0 |
| 0.5B profile + LM loss, ratings fixed (rerun of the faithful recipe) | ✅ test 0.1562 (bug version 0.1575: no effect) | Local | 3.9 h | $0 |
| GenRec-AP grid: 0.5B genres, synopsis | ✅ 0.1919, 0.1921 | Local | 2.5 h + 4.1 h | $0 |
| GenRec-AP grid: 1.5B genres, profile, synopsis | ✅ 0.1903, 0.1924, 0.1916 | Local | 6.8 + 7.1 + 9.9 h | $0 |
| GenRec-AP grid: 3B genres, profile, synopsis | ✅ 0.1973, **0.2004**, 0.1981 | Local | 13.9 + 14.2 + 19.5 h | $0 |
| 0.5B profile + LM loss, history 50 (ablation) | ❌ stopped 2026-10-12 in epoch 2 (epoch 1 checkpoint kept) | Local | ~30 h | $0 |

## Later: beyond the faithful recipe (to discuss)
Not in the paper; only after the faithful results are final, and always labeled.
- ~~Multi-position loss~~ → GenRec-AP (`--all_positions`), done: grid in [results](results.md); best 3B profile 0.2004 vs SASRec (chunked) 0.1905
- Item embeddings initialized from text (title / genres / synopsis)
- ML-25M (more items, long tail, global time split) as a second benchmark
