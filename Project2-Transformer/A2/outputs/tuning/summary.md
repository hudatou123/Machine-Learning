# Q3.4 hyperparameter tuning (local Apple M3, budget <= 1e15 FLOPs)

All runs: batch_size 32, seq_len 128, n_positions 128, warmup 100, min_lr 1e-4, grad_accum 1.

| Config | n_embd | n_head | n_layer | max_lr | steps | train FLOPs | val loss | val PPL |
|---|---|---|---|---|---|---|---|---|
| A | 256 | 4 | 4 | 5e-4   | 2400 | 9.63e14 | 5.763 | 318.34 |
| B | 256 | 4 | 4 | 1.5e-3 | 2400 | 9.63e14 | 5.797 | 329.36 |
| C | 128 | 4 | 4 | 1.5e-3 | 5500 | 9.97e14 | 5.440 | 230.50 (best) |
| D | 256 | 4 | 8 | 1.5e-3 | 2000 | 9.70e14 | 5.900 | 365.14 |
| E | 320 | 5 | 6 | 1.5e-3 | 1600 | 9.44e14 | 5.965 | 389.48 |

Final config (Q3.5): configs/final.yaml (model of C, 270300 steps x batch 64, ~9.80e16 FLOPs)
