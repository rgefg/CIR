# DeCIR Priority Experiment Results

- Run ID: `decir_priority_20260425_134129`
- Result directory: `/data2/mingyu/composed_image_retrieval/logs/decir_priority_20260425_134129`
- Source plan: `/data2/mingyu/composed_image_retrieval/decir_priority_experiments.md`
- Queue status: 16/16 jobs completed with `ok exit=0`
- GPU policy used: all 8 GPUs were idle at launch, so the runner used 6 physical GPUs: `3 4 5 6 7 0`
- Prompt policy: CIRR used `and`; CIRCO and GeneCIS used `that`
- Raw per-job results: `records/*_result.json`
- Unified run files: `PROGRESS.md`, `status.tsv`, `results.jsonl`, `queue.log`

All metric values below use the same scale as the plan tables. CIRCO `mAP@50` is reported as percentage points, so `0.2726` in raw JSON is shown as `27.26`.

## Executive Summary

Priority 1 was only partially favorable. LRDM full is best on CIRR `R_s@1`, but it does not cleanly beat naive no-SVD merging on CIRCO or GeneCIS. Under the plan's inclusion rule, this should not be used as a main-paper LRDM necessity table without more debugging or a revised claim.

Priority 2 is negative for the intended QueryDir story. DeCIR merged keeps better retrieval metrics, but QueryDir is slightly lower than retrieval-only on both CIRR and CIRCO. Under the plan's rule, do not include this QueryDir table as evidence for DeCIR.

Priority 3 documentation was completed in `docs/llm_generated_supervision.md`. The optional supervision ablation and hard-distractor analysis were not run in this batch.

## Plan Coverage

| Plan item | Status | Result / decision |
|---|---|---|
| Priority 1: LRDM necessity ablation | Done | LRDM best on CIRR `R_s@1`, but not cleanly best on CIRCO/GeneCIS. Keep internal unless revised. |
| Extra merge hyperparameter sweep, `w1*A1 + w2*A2` | Partially done | Ran no-SVD `0.75/0.25` and `0.25/0.75`. `0.25/0.75` beats LRDM on CIRCO `mAP@50`; `0.75/0.25` beats LRDM on GeneCIS Avg R@1. |
| Optional beta sensitivity, `A_r + beta A_d` | Not run | No valid result in this batch. |
| Optional retained-rank `k` sensitivity | Not run | No valid result in this batch. |
| Priority 2: composed-query direction metric | Done | Negative/weak: DeCIR merged QueryDir is slightly below retrieval-only on CIRR and CIRCO. |
| Priority 3: LLM-generated supervision documentation | Done | Documentation added with model, prompt, filtering rules, counts, and examples. |
| Optional supervision ablation | Not run | No valid result in this batch. |
| Hard distractor / edit-sensitive analysis | Not run | No valid result in this batch. |

## Priority 1: LRDM Necessity Ablation

Required table from the plan:

| Variant | Merge formula | CIRR `R_s@1` | CIRCO `mAP@50` | GeneCIS Avg. `R@1` |
|---|---|---:|---:|---:|
| Retrieval only / no transition merge | `A_m = A_r` | 64.46 | 24.92 | 16.80 |
| Average coefficient merge | `(A_r + A_d) / 2`, no SVD | 65.97 | 27.26 | 16.87 |
| Weighted no-SVD 0.75/0.25 | `0.75 A_r + 0.25 A_d` | 65.01 | 26.52 | 16.93 |
| Weighted no-SVD 0.25/0.75 | `0.25 A_r + 0.75 A_d` | 65.92 | 27.64 | 16.65 |
| LRDM full | `0.5/0.5 + truncated SVD` | 66.08 | 27.26 | 16.86 |

Detailed CIRR composed-query retrieval metrics:

| Variant | `R@1` | `R@5` | `R@10` | `R_s@1` | `R_s@2` | `R_s@3` |
|---|---:|---:|---:|---:|---:|---:|
| Retrieval only / no transition merge | 30.30 | 61.76 | 73.36 | 64.46 | 82.85 | 91.77 |
| Average coefficient merge | 31.83 | 63.00 | 74.74 | 65.97 | 83.76 | 92.06 |
| Weighted no-SVD 0.75/0.25 | 31.40 | 61.99 | 74.19 | 65.01 | 83.38 | 91.75 |
| Weighted no-SVD 0.25/0.75 | 32.19 | 63.43 | 75.41 | 65.92 | 84.02 | 92.18 |
| LRDM full | 31.98 | 62.90 | 74.79 | 66.08 | 83.76 | 91.96 |

Detailed CIRCO validation metrics:

| Variant | `mAP@5` | `mAP@10` | `mAP@25` | `mAP@50` | `R@10` |
|---|---:|---:|---:|---:|---:|
| Retrieval only / no transition merge | 19.87 | 21.91 | 24.09 | 24.92 | 50.91 |
| Average coefficient merge | 22.03 | 24.11 | 26.53 | 27.26 | 55.91 |
| Weighted no-SVD 0.75/0.25 | 21.50 | 23.50 | 25.75 | 26.52 | 54.09 |
| Weighted no-SVD 0.25/0.75 | 22.58 | 24.44 | 26.85 | 27.64 | 57.27 |
| LRDM full | 22.00 | 24.13 | 26.53 | 27.26 | 55.91 |

Detailed GeneCIS metrics:

| Variant | Avg. `R@1` | focus_attribute | change_attribute | focus_object | change_object |
|---|---:|---:|---:|---:|---:|
| Retrieval only / no transition merge | 16.80 | 21.75 | 17.14 | 13.98 | 14.34 |
| Average coefficient merge | 16.87 | 21.05 | 17.61 | 14.29 | 14.54 |
| Weighted no-SVD 0.75/0.25 | 16.93 | 21.05 | 17.76 | 14.44 | 14.49 |
| Weighted no-SVD 0.25/0.75 | 16.65 | 20.35 | 17.47 | 14.03 | 14.74 |
| LRDM full | 16.86 | 21.15 | 17.52 | 14.23 | 14.54 |

Comparison against LRDM full:

| Comparison | CIRR `R_s@1` delta | CIRCO `mAP@50` delta | GeneCIS Avg. `R@1` delta |
|---|---:|---:|---:|
| LRDM full - retrieval only | +1.63 | +2.34 | +0.06 |
| LRDM full - average merge | +0.12 | +0.00 | -0.01 |
| LRDM full - weighted no-SVD 0.75/0.25 | +1.08 | +0.74 | -0.07 |
| LRDM full - weighted no-SVD 0.25/0.75 | +0.17 | -0.38 | +0.21 |

Conclusion: LRDM is useful on CIRR and improves over retrieval-only on all three headline metrics, but the required pattern is not clean. Average merge essentially ties LRDM on CIRCO, weighted no-SVD `0.25/0.75` beats LRDM on CIRCO, and weighted no-SVD `0.75/0.25` beats LRDM on GeneCIS. Treat this as internal evidence and debug/sweep further before using it as a main LRDM necessity result.

Raw files:

- `records/p1_cirr_retrieval_only_result.json`
- `records/p1_cirr_avg_coeff_no_svd_w050_050_result.json`
- `records/p1_cirr_weighted_no_svd_w075_025_result.json`
- `records/p1_cirr_weighted_no_svd_w025_075_result.json`
- `records/p1_cirr_lrdm_full_svd_w050_050_result.json`
- `records/p1_suite_retrieval_only_result.json`
- `records/p1_suite_avg_coeff_no_svd_w050_050_result.json`
- `records/p1_suite_weighted_no_svd_w075_025_result.json`
- `records/p1_suite_weighted_no_svd_w025_075_result.json`
- `records/p1_suite_lrdm_full_svd_w050_050_result.json`

## Optional Sensitivity: Beta In `A_m = A_r + beta A_d`

This optional table was not run in this batch.

| Beta | CIRR `R_s@1` | CIRCO `mAP@50` | GeneCIS Avg. `R@1` | Status |
|---:|---:|---:|---:|---|
| 0.00 | 64.46 | 24.92 | 16.80 | Reused retrieval-only baseline |
| 0.25 | n/a | n/a | n/a | Not run |
| 0.50 | n/a | n/a | n/a | Not run |
| 0.75 | n/a | n/a | n/a | Not run |
| 1.00 | n/a | n/a | n/a | Not run |

The batch did include normalized coefficient no-SVD variants `0.75/0.25` and `0.25/0.75`; those are reported under Priority 1.

## Optional Sensitivity: Retained Rank `k`

This optional table was not run in this batch.

| Retained rank `k` | CIRR `R_s@1` | CIRCO `mAP@50` | GeneCIS Avg. `R@1` | Status |
|---|---:|---:|---:|---|
| `r/4` | n/a | n/a | n/a | Not run |
| `r/2` | n/a | n/a | n/a | Not run |
| `3r/4` | n/a | n/a | n/a | Not run |
| `r` / no denoising | See weighted no-SVD rows | See weighted no-SVD rows | See weighted no-SVD rows | Partially covered by no-SVD variants |

## Priority 2: Composed-Query Direction Metric

The metric was computed on the actual composed query generated by the retrieval model. Counts were 4181 CIRR samples and 220 CIRCO samples, with zero skipped samples.

Main comparison table:

| Model | CIRR Query-target | CIRR QueryDir | CIRCO Query-target | CIRCO QueryDir | CIRR `R_s@1` | CIRCO `mAP@50` |
|---|---:|---:|---:|---:|---:|---:|
| Retrieval only | 0.2978 | 0.2617 | 0.3292 | 0.2043 | 64.46 | 24.92 |
| Joint training | 0.2572 | 0.2118 | 0.3082 | 0.1765 | 61.09 | 21.93 |
| DeCIR merged | 0.2886 | 0.2603 | 0.3208 | 0.2033 | 66.08 | 27.26 |

Additional direction diagnostics:

| Model | Dataset | Query-source | Target-source | Delta target norm | Delta query norm |
|---|---|---:|---:|---:|---:|
| Retrieval only | CIRR | 0.2965 | 0.7951 | 0.6220 | 1.1853 |
| Joint training | CIRR | 0.2534 | 0.8618 | 0.5085 | 1.2212 |
| DeCIR merged | CIRR | 0.2872 | 0.7951 | 0.6220 | 1.1931 |
| Retrieval only | CIRCO | 0.3453 | 0.8579 | 0.5296 | 1.1437 |
| Joint training | CIRCO | 0.3190 | 0.8914 | 0.4611 | 1.1663 |
| DeCIR merged | CIRCO | 0.3367 | 0.8579 | 0.5296 | 1.1511 |

Conclusion: This diagnostic does not support the intended story. DeCIR merged improves retrieval metrics over retrieval-only, but QueryDir is slightly lower than retrieval-only on both CIRR and CIRCO. Joint training is also lower on QueryDir and retrieval. Per the plan, do not include this QueryDir table in the main paper as DeCIR evidence.

Raw files:

- `records/querydir_cirr_retrieval_only_result.json`
- `records/querydir_cirr_joint_result.json`
- `records/querydir_cirr_decir_merged_result.json`
- `records/querydir_circo_retrieval_only_result.json`
- `records/querydir_circo_joint_result.json`
- `records/querydir_circo_decir_merged_result.json`

## Priority 3: LLM-Generated Supervision Documentation

Status: completed as documentation, no GPU experiment.

Documentation file:

- `/data2/mingyu/composed_image_retrieval/docs/llm_generated_supervision.md`

Recorded details:

- LLM: `glm-4.7-flash` through ZhipuAI batch `/v4/chat/completions`
- Generation setting: reasoning/thinking enabled, `temperature=0.7`
- Forward prompt template: recorded in the documentation file
- Filtering rules: recorded in the documentation file
- Raw parsed forward generations: 727,573
- First-stage kept generations: 667,229, 91.71%
- Merged candidate rows: 2,725,236
- Retrieval-cleaned rows: 2,598,571, 95.35%
- Rows with reverse supervision: 1,277,145
- Example generated tuples: 5 examples recorded

Example generated tuples:

```jsonl
{"id":"000003390","instruction":"Change the man's attire to a business suit.","modified_caption":"A man in a business suit conducts business during an event.","reverse_instruction":"Change the man's attire to non-business suit attire"}
{"id":"000007547","instruction":"Change the rehearsal setting to take place at night.","modified_caption":"Dancers rehearse for a performance at night.","reverse_instruction":"Change the rehearsal setting to day time."}
{"id":"000020829","instruction":"Change the time of day from 'a week' to 'on a sunny afternoon'.","modified_caption":"actor arrives on a sunny afternoon to a screening","reverse_instruction":"Change the time of day to 'a week'."}
{"id":"000015554","instruction":"Change the color of the brick wall from red to blue.","modified_caption":"a blue brick wall from a building","reverse_instruction":"Change the color of the brick wall to red"}
{"id":"000018986","instruction":"Change the color of the patterned piece from blue and white to green and black.","modified_caption":"In the sea, she showcased her flat stomach in a green and black patterned piece as she took to the sea for a paddle.","reverse_instruction":"Change the color of the patterned piece from green and black to blue and white"}
```

Manual audit over 200 sampled tuples was listed as optional in the plan and was not run.

## Optional Supervision Ablation

This optional training/evaluation ablation was not run in this batch.

| Supervision | CIRR `R_s@1` | CIRCO `mAP@50` | QueryDir / branch diagnostic | Status |
|---|---:|---:|---:|---|
| Retrieval only / no transition | 64.46 | 24.92 | See QueryDir retrieval-only rows | Reused baseline |
| Forward only | n/a | n/a | n/a | Not run |
| Forward + reverse | n/a | n/a | n/a | Not run |
| Forward + reverse + inverse consistency | n/a | n/a | n/a | Not run |

## Hard Distractor / Edit-Sensitive Analysis

This internal diagnostic was requested in the plan text but was not run in this batch. There is no valid result yet for the claim that endpoint-only retrieval fails on hard distractors or target-attribute shortcuts.

Recommended next implementation if needed:

- Build a hard subset where candidates share target attributes but conflict with source-preserving attributes.
- Compare retrieval-only, joint training, and DeCIR merged on the same hard subset.
- Report the hard-subset delta separately from global retrieval metrics.

## Final Recommendation For Paper Use

Use the LLM supervision documentation immediately for reproducibility/checklist material.

Do not use the current QueryDir table as final DeCIR evidence.

Do not use the current LRDM necessity table as a clean main-paper table unless the claim is narrowed to CIRR or additional sweeps recover a consistent LRDM advantage on CIRCO/GeneCIS.
