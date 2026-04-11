# Shared-B CIRR Ablation

All results below use the same `step1400` checkpoint pair from:

- retrieval: `epoch_0_step_1400.pt`
- geo: `epoch_0_step_1400_geo_lora_ema.pt`

Dataset / prompt:

- `CIRR`
- train prompt: `and`
- eval prompt: `and`

Comparison target:

- vary the number of shallow layers using `Shared-B + SVD-denoise(A)` merge
- deeper layers use `TIES`
- `0` means plain `TIES`
- `12` means full-layer `Shared-B SVD-denoise(A)` without deep `TIES`

## Step1400 Results

| Shallow layers | Merge rule | R@1 | R@10 | R@50 | R_subset@1 | R_subset@2 | R_subset@3 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | plain TIES | 28.87 | 74.07 | 93.16 | 61.61 | 81.20 | 90.89 |
| 2 | shallow SVD-denoise, deep TIES | 29.06 | 74.07 | 93.16 | 62.40 | 81.80 | 91.25 |
| 4 | shallow SVD-denoise, deep TIES | 28.82 | 73.83 | 93.04 | 62.78 | 81.97 | 91.20 |
| 6 | shallow SVD-denoise, deep TIES | 28.61 | 73.45 | 92.85 | 62.71 | 81.97 | 91.01 |
| 8 | shallow SVD-denoise, deep TIES | 27.77 | 72.38 | 92.56 | 61.64 | 81.18 | 90.36 |
| 10 | shallow SVD-denoise, deep TIES | 27.48 | 72.11 | 92.54 | 61.35 | 81.13 | 90.34 |
| 12 | full-layer SVD-denoise | 27.48 | 72.64 | 92.56 | 61.37 | 81.15 | 90.58 |

## Takeaways

- Full-gallery retrieval (`R@1/R@10/R@50`) peaks at `2` shallow layers.
- Subset retrieval (`R_subset@1/R_subset@2/R_subset@3`) peaks at `4` shallow layers.
- The half-split setting (`6`) is strong on subset metrics, but not the best point overall.
- Moving too deep into non-shared layers (`8/10/12`) causes a clear drop.

## ViT-L Step1400 Results

All results below use the same `step1400` checkpoint pair from:

- retrieval: `epoch_0_step_1400.pt`
- geo: `epoch_0_step_1400_geo_lora_ema.pt`

Experiment setup:

- `ViT-L/14 + SEARLE Phi`
- `Shared-B` enabled for all 12 layers during training
- `geo` loss uses `fwd + rev + zero` (no reverse-consistency term)
- merge-time ablation varies how many shallow layers use `SVD-denoise(A)` before deep-layer `TIES`

| Shallow layers | Merge rule | R@1 | R@5 | R@10 | R@50 | R_subset@1 | R_subset@2 | R_subset@3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain TIES | 30.47 | 62.62 | 74.41 | 92.01 | 63.29 | 83.21 | 91.65 |
| 2 | shallow SVD-denoise, deep TIES | 31.43 | 62.71 | 74.81 | 92.27 | 64.46 | 83.62 | 92.04 |
| 4 | shallow SVD-denoise, deep TIES | 31.69 | 62.86 | 74.93 | 92.32 | 64.98 | 83.47 | 91.99 |
| 6 | shallow SVD-denoise, deep TIES | 31.79 | 63.19 | 74.98 | 92.23 | 64.65 | 83.50 | 92.13 |
| 8 | shallow SVD-denoise, deep TIES | 31.60 | 63.19 | 74.93 | 92.27 | 64.84 | 83.59 | 91.99 |
| 10 | shallow SVD-denoise, deep TIES | 31.86 | 62.90 | 74.89 | 92.37 | 65.25 | 83.52 | 92.08 |
| 12 | full-layer SVD-denoise | 31.98 | 62.90 | 74.79 | 92.39 | 66.08 | 83.76 | 91.96 |

### ViT-L Takeaways

- If the story is about **subset discrimination under full Shared-B**, `R_subset@1` is the strongest supporting metric: it improves steadily from `63.29` (`l0`) to `66.08` (`l12`).
- If the story is about an **increase-then-decrease trend**, `R@10` is the cleanest metric: `74.41 -> 74.81 -> 74.93 -> 74.98 -> 74.93 -> 74.89 -> 74.79`, peaking at `l6`.
- `R@1`, `R@50`, and `R_subset@1` do not support a half-split-best story for `ViT-L`; they prefer deeper or full Shared-B.
