# Geo Branch CIRR Proxy Summary

Date: 2026-04-18

## Protocol

- Backbone: `ViT-L/14 + SEARLE Phi`
- Dataset/eval: `CIRR val`
- Merge: plain `TIES`, `0.5:0.5`, `density=0.9`, `text-only`
- Proxy budget: `150` train steps, evaluate `step150`
- Shared-B: `12 layers`, retrieval-only shared-B update enabled
- Prompt: `and`
- Reverse consistency term: disabled globally in this sweep (`geo_reverse_weight=0.0`)

## Q1: Make Geo Branch Less Split From The Visual Path

### Baseline

- Geo source anchor: pure text source caption
- Loss: `fwd + rev + zero`

### Visual-conditioned candidate

- Geo source anchor: `blend(text_source, image_anchor)`
- Image anchor: `ref_img -> visual encoder -> img2text -> "a photo of *"`
- Blend weight: `0.25`
- Anchor detach: `True`
- Loss: `fwd + rev + zero`

### Results

| setting | R@1 | R@5 | R@10 | R@50 | R_subset@1 | R_subset@2 | R_subset@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline text source | 25.64 | 56.30 | 70.32 | 90.91 | 56.25 | 77.18 | 88.73 |
| visual blend 0.25 detach | 25.88 | 56.54 | 70.25 | 90.77 | 56.49 | 77.25 | 88.93 |

### Interim take

- The visual-conditioned geo branch is viable.
- It uses the same `visual encoder + img2text` path as retrieval, so the method figure is no longer fully split.
- On this proxy, it is effectively performance-neutral:
  - `R@1`: `+0.24`
  - `R@5`: `+0.24`
  - `R@10`: `-0.07`
  - `R@50`: `-0.14`
  - `R_subset@1`: `+0.24`

## Q2: Geo-loss Term Ablation

Setup:

- `full = fwd + rev + zero`
- `drop-rev = fwd + zero`
- `drop-zero = fwd + rev`
- `fwd-only`

### Results

| loss setting | R@1 | R@5 | R@10 | R@50 | R_subset@1 | R_subset@2 | R_subset@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full (`fwd + rev + zero`) | 25.64 | 56.30 | 70.32 | 90.91 | 56.25 | 77.18 | 88.73 |
| drop-rev (`fwd + zero`) | 25.42 | 55.66 | 69.41 | 90.58 | 55.47 | 76.80 | 88.40 |
| drop-zero (`fwd + rev`) | 25.19 | 55.75 | 69.55 | 90.50 | 55.39 | 76.80 | 88.28 |
| fwd-only | 25.33 | 55.68 | 69.29 | 90.60 | 55.49 | 76.75 | 88.45 |

### Take

- `full` is the best setting on every composed metric in this proxy.
- Removing either `rev` or `zero` alone causes only a modest drop:
  - roughly `-0.2 to -0.45` on `R@1`
  - roughly `-0.75 to -0.91` on `R@10`
  - roughly `-0.31 to -0.41` on `R_subset@1`
- `fwd-only` is still viable, but it is also below `full` on every metric.
- The pattern is:
  - `fwd` is the core signal.
  - `rev` and `zero` are both helpful, but neither looks individually indispensable.
  - `rev` and `zero` appear partially redundant, because removing either one produces a similar drop.

### Practical answer

- If the goal is the strongest default training recipe, keep all three: `fwd + rev + zero`.
- If the goal is a simpler story, `fwd` can be treated as the essential term, while `rev` and `zero` can be described as auxiliary regularizers that give small but consistent gains.
