# FashionIQ / GeneCIS Tuning Plan

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

## Next Experiments

1. `No-dropout`, short run for overfitting check.
   - Future FashionIQ/GeneCIS experiments only run `1000 step`.
   - Save from `step 200`, then every `200 step`.
   - Disable watcher during training; after training finishes, use the same two training GPUs to run standalone and merged evaluation.
   - Goal: test whether current degradation mainly comes from late-stage overfitting.

2. Prompt alignment experiment.
   - During training, duplicate the instruction once so the training prompt matches the evaluation-style `a and b` caption format.
   - Keep other settings unchanged and run one `1000 step` experiment.
   - Goal: test whether the train/eval prompt mismatch is hurting FashionIQ.

3. Logit scale experiment.
   - Turn off `reset logit scale`, keep the rest of the setup unchanged, and run one `1000 step` experiment.
   - Goal: test whether current logit scale is too small and limiting retrieval transfer to FashionIQ/GeneCIS.
