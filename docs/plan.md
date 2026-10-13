# Plan

Remaining work. Overview → [README](../README.md) · results → [results.md](results.md)

## Remaining
| Task | Status | Machine | Time | Cost |
|---|---|---|---|---|
| **Final: 7B profile + LM loss** (full recipe) | ⬜ ready | Cloud A100 80GB | ~9–10 h | ~$15–16 |
| 0.5B profile + LM loss, history 50 (ablation) | 🔄 running | Local | ~11 h | $0 |

## Later: beyond the faithful recipe (to discuss)
Not in the paper; only after the faithful results are final, and always labeled.
- Multi-position loss (grade every movie in a prompt; ~16–20× more training signal per compute)
- Item embeddings initialized from text (title / genres / synopsis)
- ML-25M (more items, long tail, global time split) as a second benchmark
