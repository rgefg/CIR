# FashionIQ / GeneCIS Tuning Plan
tips:1.一旦实验失败了，你需要删除该实验的目录，不要留着太多垃圾临时目录了。
2.每隔30min找卡，只有没有进程的空闲卡才能用，找到2卡空闲即可开始实验一，实验1稳定运行后你就可以停止工作了，把控制交还。
## Current Record

- `No-drop`, `raw base + ema geo`, `step 800`, merged result:
  - FashionIQ `dress R@10 = 23.55`
  - FashionIQ `shirt R@10 = 30.91`
  - FashionIQ `toptee R@10 = 30.90`
  - GeneCIS `focus_attribute R@1 = 19.70`
  - GeneCIS `change_attribute R@1 = 16.71`
  - GeneCIS `focus_object R@1 = 14.74`
  - GeneCIS `change_object R@1 = 17.04`

- Old reference, `dropout=0.5`, merged best result:
  - FashionIQ `dress R@10 = 24.34`
  - FashionIQ `shirt R@10 = 30.27`
  - FashionIQ `toptee R@10 = 30.90`
  - GeneCIS `focus_attribute R@1 = 19.90`
  - GeneCIS `change_attribute R@1 = 16.76`
  - GeneCIS `focus_object R@1 = 13.47`
  - GeneCIS `change_object R@1 = 14.80`

- `ExpA Shared-A`, `step 600`, merge ablation:
  - Existing result `B-sum + shared A`:
    - FashionIQ `dress/shirt/toptee R@10 = 22.91 / 28.61 / 29.07`
    - GeneCIS `focus_attribute/change_attribute/focus_object/change_object R@1 = 19.85 / 16.34 / 14.23 / 16.02`
  - `Delta-TIES` on dense delta matrix:
    - FashionIQ `dress/shirt/toptee R@10 = 23.25 / 26.55 / 27.79`
    - GeneCIS `focus_attribute/change_attribute/focus_object/change_object R@1 = 19.50 / 16.24 / 13.93 / 15.00`
  - `B-TIES + shared A`:
    - FashionIQ `dress/shirt/toptee R@10 = 23.00 / 26.45 / 27.89`
    - GeneCIS `focus_attribute/change_attribute/focus_object/change_object R@1 = 19.75 / 16.38 / 13.98 / 15.05`
  - Conclusion:
    - On current `ExpA step600`, simple `B-sum + shared A` is still the best merge rule.
    - `Delta-TIES` and `B-TIES` do not improve the Shared-A route.

- `ExpA Shared-A RetAOnly`, retrieval-only `A` update:
  - `raw + raw` merge:
    - `step 600`: FashionIQ `dress/shirt/toptee R@10 = 21.27 / 24.34 / 25.04`
    - `step 600`: GeneCIS `focus_attribute/change_attribute/focus_object/change_object R@1 = 18.75 / 14.82 / 11.17 / 12.50`
    - `step 800`: FashionIQ `dress/shirt/toptee R@10 = 20.87 / 24.48 / 25.19`
    - `step 800`: GeneCIS `focus_attribute/change_attribute/focus_object/change_object R@1 = 18.90 / 15.15 / 10.97 / 12.45`
  - `raw + geo ema` merge:
    - `step 600`: FashionIQ `dress/shirt/toptee R@10 = 23.65 / 28.95 / 29.12`
    - `step 600`: GeneCIS `focus_attribute/change_attribute/focus_object/change_object R@1 = 19.55 / 16.05 / 13.72 / 16.28`
    - `step 800`: FashionIQ `dress/shirt/toptee R@10 = 23.15 / 28.46 / 28.81`
    - `step 800`: GeneCIS `focus_attribute/change_attribute/focus_object/change_object R@1 = 19.90 / 15.96 / 13.98 / 15.10`
  - Conclusion:
    - Limiting `A` updates to retrieval only helps relative to `raw + raw`, but does not beat the old non-Shared-A `raw base + geo ema` baseline.
    - Best point in this experiment is `step 600 raw + geo ema`, and it is still below `Exp1 no-drop`.

- `Hybrid Shared-A Final`, shallow `0-5` Shared-A + `B-sum`, deep `6-11` TIES:
  - `CIRR` merged:
    - `step 600`: `R@1/R@10/R@50 = 31.81 / 74.55 / 91.58`
    - `step 800`: `31.81 / 74.91 / 91.84`
    - `step 1000`: `32.22 / 75.01 / 91.70`
    - `step 1200`: `31.88 / 74.58 / 91.60`
    - `step 1400`: `31.93 / 74.43 / 91.41`
    - `step 1600/final`: `31.64 / 74.03 / 91.13`
    - Best `CIRR` point: `step 1000`
  - `FashionIQ + GeneCIS` merged:
    - `step 600`: FashionIQ `dress/shirt/toptee R@10 = 23.15 / 28.46 / 29.07`
    - `step 600`: GeneCIS `focus_attribute/change_attribute/focus_object/change_object R@1 = 19.25 / 16.38 / 14.39 / 17.04`
    - `step 800`: `22.91 / 27.97 / 28.35` and `19.20 / 16.29 / 14.95 / 15.92`
    - `step 1000`: `22.71 / 27.18 / 27.38` and `19.00 / 16.19 / 14.80 / 15.66`
    - `step 1200`: `22.41 / 26.59 / 27.49` and `19.00 / 15.96 / 14.49 / 15.71`
    - `step 1400`: `22.01 / 26.59 / 27.33` and `18.85 / 15.77 / 14.08 / 15.71`
    - `step 1600/final`: `21.76 / 26.50 / 26.93` and `19.00 / 15.72 / 13.83 / 15.31`
    - Best `FashionIQ/GeneCIS` point: `step 600`
  - Conclusion:
    - This hybrid design does not collapse training, but merged downstream metrics decay after the early stage.
    - It does not beat the old `Exp1 no-drop, raw base + geo ema` baseline on `FashionIQ/GeneCIS`.

- `Hybrid Shared-A`, `instruction dropout = 0.2`, `1000 step`, `FashionIQ-only merged`:
  - Settings:
    - Same hybrid layerwise merge rule as above.
    - Only merged `FashionIQ` was scheduled, no standalone eval.
    - Training completed normally; evaluation was stopped early after `step 800` because the trend was already weak.
  - `raw + geo ema` merged:
    - `step 600`: FashionIQ `dress/shirt/toptee R@10 = 22.11 / 27.67 / 27.54`
    - `step 800`: FashionIQ `dress/shirt/toptee R@10 = 21.27 / 27.72 / 27.13`
  - Conclusion:
    - `instruction dropout = 0.2` does not help this hybrid route.
    - It is already below the no-drop hybrid result by `step 600`, and continues to soften by `step 800`.

- Prompt alignment follow-up:
  - Old `duplicate-and` / duplicated-instruction training prompt did not beat the old `Exp1 no-drop, raw + geo ema` baseline, so `a and a` style duplication is not a useful direction for now.
  - New `that`-aligned training prompt, `1000 step`, `FashionIQ-only merged`:
    - `TIES`
      - `step 600`: FashionIQ `dress/shirt/toptee R@10 = 23.95 / 29.93 / 30.95`
      - `step 800`: `24.00 / 29.88 / 31.00`
      - `step 1000/final`: `24.10 / 29.24 / 30.49`
    - `Hybrid`
      - `step 600`: `24.24 / 29.74 / 30.60`
      - `step 800`: `24.10 / 29.54 / 30.55`
      - `step 1000/final`: `24.19 / 29.29 / 30.60`
  - Conclusion:
    - Aligning the connector word from training `and` to evaluation-style `that` is useful.
    - Best overall point in this run is `TIES step 800`.
    - This improves `dress` and `toptee` over the old `and`-prompt run, but `shirt` still does not exceed the old no-drop best.

- Current judgment:
  - High `instruction dropout = 0.5` is not a good default for this line; it is not the best overall setting even if a few single metrics look slightly better.
  - Prompt alignment is not uniformly useless: `a and a` duplication looks unhelpful, but simple `that`-style alignment does help.

## Next Experiments
common settings：Future FashionIQ/GeneCIS experiments only run `1000 step`.
   - Save from `step 200`, then every `200 step`.
   - Disable watcher during training; after training finishes, use the same two training GPUs to run standalone and merged evaluation. for other settings ,refer to /data2/mingyu/composed_image_retrieval/train_with_dropout.sh

1. `No-dropout`, short run for overfitting check.
   - Goal: test whether current degradation mainly comes from late-stage overfitting.

2. Prompt alignment experiment.
   - During training, duplicate the instruction once so the training prompt matches the evaluation-style `a and b` caption format.
   - Keep other settings unchanged and run one `1000 step` experiment.
   - Goal: test whether the train/eval prompt mismatch is hurting FashionIQ.

3. Logit scale experiment.
   - Turn off `reset logit scale`, keep the rest of the setup unchanged, and run one `1000 step` experiment.
   - Goal: test whether current logit scale is too small and limiting retrieval transfer to FashionIQ/GeneCIS.

4. Layer-wise merge experiment from GC probe.
   - Motivation: current GC probe suggests shallow text blocks have lower conflict, while deeper blocks have higher conflict.
   - Proposed merge rule:
     - layers `0-5`: simple weighted average / `B-sum`
     - layers `6-11`: conservative merge such as `TIES`
   - Simple A/B:
     - A: all 12 layers use the same global merge rule
     - B: layers `0-5` use average, layers `6-11` use `TIES`
   - Goal: test whether conflict-aware layer-wise merge is better than one global merge rule.
