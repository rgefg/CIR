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
