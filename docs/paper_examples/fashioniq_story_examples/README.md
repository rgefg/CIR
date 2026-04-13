## FashionIQ Story Examples

This folder contains qualitative `FashionIQ val` cases selected to support the retrieval-story analysis.

Selection rule:

- use the `ViT-L/14 + SEARLE Phi` `step1000` merged run,
- search for cases where the merged model retrieves the correct target at rank 1,
- while `Pic2Word`, the retrieval branch, and the geo branch all fail on the full gallery.

The strongest case is:

- [shirt / case_08_shirt_step1000](/data2/mingyu/composed_image_retrieval/docs/paper_examples/fashioniq_story_examples_shirt/case_08_shirt_step1000)

Why this case matters:

- `Pic2Word` returns the reference image, so it is strongly source-biased.
- `geo` also returns the reference image.
- the retrieval branch follows the edit instruction, but retrieves the wrong edited shirt.
- the merged model retrieves the correct target image.

This is the cleanest qualitative example of:

- source bias in `Pic2Word`,
- edit-following but identity-confused retrieval behavior,
- and corrective behavior after merging retrieval and geo branches.

Backup case:

- [dress / case_01_dress_step1000](/data2/mingyu/composed_image_retrieval/docs/paper_examples/fashioniq_story_examples_dress/case_01_dress_step1000)

Note:

- the old `fashioniq_oracle_examples` folder was built for candidate-set probes, not for retrieval ranking visualization.
- the folders here are the ones intended for paper figures: each case stores the reference image, the ground-truth target, and the top-1 result from each model branch.
