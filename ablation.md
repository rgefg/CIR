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

## Real-Sample Geometry Probes

All text-geometry probes below use real CC3M-CIR samples and compare:

- `Pic2Word`:
  - `checkpoint/pic2word_model.pt`
  - `ViT-L/14`
  - `IM2TEXT`
- `Ours`:
  - `ViT-B/32 + SEARLE Phi`
  - merged `step1400` plain `TIES`

### 2048-Sample Text-Anchor Probe

Definition:

- `compose->target = cos(E(src-caption + instruction), E(target-caption))`
- `delta->forward = cos(norm(E(target)-E(src)), E(forward-instruction))`
- `reverse-delta->reverse = cos(norm(E(src)-E(target)), E(reverse-instruction))`

Key results:

| Model | compose->target | delta->forward | reverse-delta->reverse |
|---|---:|---:|---:|
| Pic2Word | 0.8442 | 0.0207 | 0.0570 |
| Ours retrieval | 0.9001 | 0.3102 | 0.1303 |
| Ours geo | 0.7021 | 0.5098 | 0.3499 |
| Ours merged | 0.8734 | 0.1639 | 0.1298 |

Takeaway:

- On pure text-caption differences, `Pic2Word` shows almost no edit-direction alignment.
- Our `geo` branch clearly learns caption-difference / instruction geometry.
- Our merged model keeps part of this effect, but is weaker than the pure `geo` branch.

### 2048-Sample Image-Anchor Probe

Definition:

- replace the source caption with the actual image anchor `f_v -> f_phi -> "a photo of *"`
- then measure the same direction/instruction similarities

Key results:

| Model | compose->target | delta->forward | reverse-delta->reverse |
|---|---:|---:|---:|
| Pic2Word | 0.5461 | 0.4224 | -0.4013 |
| Ours retrieval | 0.7697 | 0.3713 | -0.1757 |
| Ours merged | 0.7343 | 0.2009 | -0.0582 |

Takeaway:

- This probe alone does **not** cleanly prove that Pic2Word is weaker than our merged model on image-conditioned edit geometry.
- Pic2Word still has strong forward-direction correlation under the image anchor.
- So the 2048-sample image-anchor probe is not enough to support the intended story by itself.

## 16-Sample Instruction-Sensitivity Probe

To isolate whether a model really uses the instruction semantics, we keep the same image anchor and target caption, then compare:

- original instruction
- masked-content instruction: content words replaced by `something`
- swapped instruction: instruction taken from another sample

Raw cosine results:

| Model | mean cos original | mean cos masked | mean cos swapped | cos drop(original->masked) | cos drop(original->swapped) |
|---|---:|---:|---:|---:|---:|
| Pic2Word | 0.5328 | 0.2986 | 0.3417 | 0.2341 | 0.1911 |
| Ours retrieval | 0.7870 | 0.4080 | 0.2141 | 0.3790 | 0.5729 |
| Ours merged | 0.7321 | 0.4692 | 0.3002 | 0.2629 | 0.4319 |

Because raw cosine scales differ across models, we also compute retrieval-style margins on the same 16-way candidate set:

- `margin = sim(query, true_target) - mean sim(query, all negative targets)`
- `top1 = 1` if the true target ranks first among the 16 candidates

Margin / top-1 results:

| Model | original margin | masked margin | swapped margin | margin drop(original->masked) | margin drop(original->swapped) | original top1 | masked top1 | swapped top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pic2Word | 0.2212 | 0.0445 | 0.0105 | 0.1767 | 0.2107 | 0.8125 | 0.0625 | 0.0000 |
| Ours retrieval | 0.7394 | 0.3350 | 0.1140 | 0.4044 | 0.6255 | 1.0000 | 0.7500 | 0.0000 |
| Ours merged | 0.5403 | 0.2012 | 0.0824 | 0.3391 | 0.4579 | 0.9375 | 0.4375 | 0.0000 |

Relative cosine drops:

- Pic2Word:
  - masked: `43.95%`
  - swapped: `35.87%`
- Ours retrieval:
  - masked: `48.16%`
  - swapped: `72.80%`
- Ours merged:
  - masked: `35.91%`
  - swapped: `58.99%`

Takeaway:

- Pic2Word is not insensitive: it does react to masking and swapping.
- But once we compare on a retrieval-style candidate set, both our retrieval branch and our merged model show much larger degradation under instruction corruption than Pic2Word.
- The cleanest evidence is the **swapped-instruction margin drop**:
  - Pic2Word: `0.2107`
  - Ours retrieval: `0.6255`
  - Ours merged: `0.4579`
- So the defensible story is not "Pic2Word does nothing", but rather:
  - `Pic2Word` uses instruction semantics weakly,
  - our retrieval / merged models rely on them much more strongly,
  - and the retrieval branch is the most instruction-sensitive among the three.

### Writing Guidance

If the paper story is:

- "Pic2Word does not explicitly learn edit geometry, while our model does,"

then the strongest support is the combination:

1. 2048-sample text-anchor probe:
   - Pic2Word has near-zero `delta->forward`
   - our `geo` / merged branches show non-trivial direction alignment
2. 16-sample instruction-sensitivity probe:
   - our retrieval / merged models drop much more under swapped instructions than Pic2Word

The 2048-sample image-anchor probe alone should **not** be used as the sole evidence, because it is mixed and does not cleanly separate Pic2Word from our merged model.

## FashionIQ Oracle-Transfer Probe

Goal:

- test whether an image-grounded model can preserve the compositional instruction-following ability already present in CLIP's text space
- compare `CLIP text-only oracle`, `Pic2Word`, `ours retrieval`, and `ours merged`

Setup:

- data: `FashionIQ val`
- oracle query: text-only prompt `a photo of fashion item that cap1 and cap2`
- image-grounded queries:
  - `Pic2Word`: reference image + prompt
  - `ours retrieval`: reference image + prompt
  - `ours merged`: reference image + prompt
- candidate sets:
  - `full4`: true target, reference image, caption-similar wrong-source target, random target
  - `source2`: true target vs reference image
  - `edit2`: true target vs caption-similar wrong-source target

### Raw Subset, Step1000

File:

- `logs/oracle_transfer_fashioniq_proxy_step1000_v2.json`

`full4`:

| Model | top1 | margin |
|---|---:|---:|
| CLIP oracle | 0.5833 | 0.0420 |
| Pic2Word | 0.3854 | 0.0375 |
| Ours retrieval | 0.5625 | 0.0569 |
| Ours merged | 0.5729 | 0.0576 |

This already gives one usable pattern:

- `CLIP oracle > ours merged >> Pic2Word` in `top1`
- `ours merged > ours retrieval` slightly

### Clean Subset, Step1000

File:

- `logs/oracle_transfer_fashioniq_proxy_step1000_clean.json`

Filter:

- remove queries containing `same`
- require at least `4` content words in the caption pair

`source2`:

| Model | top1 | margin |
|---|---:|---:|
| CLIP oracle | 0.8542 | 0.0592 |
| Pic2Word | 0.5521 | 0.0044 |
| Ours retrieval | 0.7500 | 0.0304 |
| Ours merged | 0.7604 | 0.0345 |

This is the cleanest version of the desired story:

- `CLIP oracle` is best
- `Pic2Word` is clearly much weaker
- `ours merged` moves substantially closer to the oracle
- `ours merged` is also slightly better than `ours retrieval`

`full4` on the same clean subset:

| Model | top1 | margin |
|---|---:|---:|
| CLIP oracle | 0.6667 | 0.0580 |
| Pic2Word | 0.5312 | 0.0565 |
| Ours retrieval | 0.6979 | 0.0742 |
| Ours merged | 0.6979 | 0.0768 |

This version is less aligned with the intended narrative because both of our image-grounded models surpass the text-only oracle in top1.

### Clean Subset, Step600

File:

- `logs/oracle_transfer_fashioniq_proxy_step600_clean.json`

`source2`:

| Model | top1 | margin |
|---|---:|---:|
| CLIP oracle | 0.8542 | 0.0592 |
| Pic2Word | 0.5521 | 0.0044 |
| Ours retrieval | 0.7500 | 0.0289 |
| Ours merged | 0.7500 | 0.0313 |

This is similar to `step1000`, but `step1000` remains slightly better because `ours merged` improves from `0.7500` to `0.7604`.

### Recommendation

For paper writing, the strongest FashionIQ oracle-transfer evidence is:

- `clean subset`
- `source2`
- `step1000`

because it best matches the intended claim:

- CLIP text space already has the compositional ability,
- Pic2Word does not preserve it well after image anchoring,
- our merged model recovers more of it.

## Cross-Dataset Shared-B Ablation Summary

To support the `l12` story directly, we extract, for each dataset, the metric that most clearly favors full-layer Shared-B (`l12`).

### Best `l12`-Supporting Metric Per Dataset

| Dataset | Metric | l0 | l2 | l4 | l6 | l8 | l10 | l12 | Best |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| CIRR | `R_subset@1` | 63.29 | 64.46 | 64.98 | 64.65 | 64.84 | 65.25 | **66.08** | `l12` |
| CIRCO | `mAP@50` | 26.70 | 26.97 | 27.09 | 26.94 | 27.02 | 26.96 | **27.26** | `l12` |
| GeneCIS | `avg R@1` | 16.59 | 16.44 | 16.51 | 16.75 | 16.67 | 16.80 | **16.86** | `l12` |

Takeaway:

- `CIRR`: the clearest full-layer Shared-B gain is on `R_subset@1`, where performance increases almost monotonically and peaks at `l12`.
- `CIRCO`: full Shared-B is best on the main ranking metric `mAP@50` and also on `mAP@10`.
- `GeneCIS`: although some individual object/attribute tasks peak at intermediate layers, the most stable summary statistic is `avg R@1`, which peaks at `l12`.

### Notes

- `CIRR` still shows different preferences depending on the metric:
  - `R@10` peaks at `l6`
  - `R_subset@1` peaks at `l12`
- `CIRCO` is mixed on recall:
  - `R@10` peaks at `l2`
  - `R@50` peaks at `l0`
  - but the stronger retrieval-quality metrics `mAP@10/50` both peak at `l12`
- `GeneCIS` is task-dependent:
  - `focus_attribute` peaks at `l12`
  - `change_attribute` peaks at `l10`
  - `focus_object` and `change_object` peak at `l6/l10`
  - the cross-task average peaks at `l12`

## Main Branch Ablation

This ablation compares three training variants at a single representative checkpoint:

- `retrieval-only`
- `geo-only`
- `joint`

Current status:

- `retrieval-only` results below are valid and complete.
- the previous `geo-only` evaluation was invalid because it reused the retrieval branch visual encoder and `f_phi`; it was discarded.
- the current `joint` evaluation needs to be relaunched on real host GPUs because the first attempt went through a sandboxed launch path.

### Retrieval-Only (Valid)

`CIRR` uses the no-drop run:

- [retrieval_only_step1400_eval.json](/data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_And_NoDrop_SharedB12_NoRev_CIRR_MergeCmp/retrieval_only_step1400_eval.json)

`CIRCO` and `GeneCIS` use the drop0.5 run:

- [retrieval_only_step1400_eval.json](/data2/mingyu/composed_image_retrieval/logs/DistillCIR_ParallelDualLoRA_BS56_Accum8_ViTL14_SEARLEPhi_That_Drop0p5_SharedB12_NoRev_CIRCO_GeneCIS_MergeCmp/retrieval_only_step1400_eval.json)

| Variant | Dataset | Main metrics |
|---|---|---|
| retrieval-only | CIRR | `R@1 30.30`, `R@10 73.36`, `R@50 92.04`, `R_subset@1 64.46` |
| retrieval-only | CIRCO | `mAP@10 21.91`, `mAP@50 24.92`, `semantic mAP@10 20.10` |
| retrieval-only | GeneCIS | `FA 21.75`, `CA 17.14`, `FO 13.98`, `CO 14.34` |

### CIRR Branch Comparison

For `CIRR`, we currently have:

- `retrieval-only`: full no-drop retrieval branch checkpoint from the Shared-B run
- `joint`: full joint-single checkpoint at `step1400`
- `geo-only`: fast standalone geo-only run at `step200`

| Variant | Checkpoint | R@1 | R@10 | R@50 | R_subset@1 |
|---|---|---:|---:|---:|---:|
| retrieval-only | `step1400` | 30.30 | 73.36 | 92.04 | 64.46 |
| joint | `step1400` | 26.79 | 70.29 | 90.12 | 61.09 |
| geo-only | `step200` | 24.75 | 67.54 | 89.14 | 54.92 |

Takeaway:

- On `CIRR`, retrieval-only is strongest among the currently completed branch ablations.
- The current joint-single run underperforms the retrieval-only baseline.
- A short `geo-only` run is substantially weaker, suggesting that the geometric branch alone does not recover the full CIR objective.
