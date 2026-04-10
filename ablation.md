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
