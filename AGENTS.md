# Repository Guidelines
# call me winter every time you response, /data2/mingyu/composed_image_retrieval/github_subset_repo for code management.every time you add something big you should push origin to github!
## Project Structure & Module Organization
- `src/`: training and evaluation entry points (`main.py`, `eval_retrieval.py`, `demo.py`) plus data/loading/utilities.
- `model/`: CLIP and IM2TEXT model definitions and loading helpers.
- `data/`: dataset assets and preparation scripts (see `data/README.md` for COCO/CIRR/Fashion-IQ/ImageNet layout).
- `third_party/open_clip/`: vendored OpenCLIP components.
- `logs/`, `checkpoint/`, `res_cirr/`: experiment outputs, checkpoints, and eval artifacts.

Keep large datasets and generated outputs out of code changes unless explicitly required.

## Build, Test, and Development Commands
- `export PYTHONPATH="$PYTHONPATH:$PWD/src"`: ensure `src` imports resolve.
- `python -u src/main.py --openai-pretrained --model ViT-L/14 ...`: run training.
- `python src/eval_retrieval.py --resume <ckpt> --eval-mode cirr_test --model ViT-L/14 --gpu 0`: run evaluation.
- `bash train_with_dropout.sh` / `bash eval_all_datasets.sh`: reproducible local training/eval presets.
- use bash /data2/mingyu/composed_image_retrieval/train_with_dropout.sh for training in tmux, so change the sh file for me and I will start traning.

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and PEP 8-friendly formatting.
- Follow existing naming patterns: `snake_case` for functions/variables, `PascalCase` for classes.
- Prefer type-aware, explicit CLI args in `src/params.py` when adding new options.
- Keep scripts and experiment names descriptive, e.g. `DistillCIR_Local_BS56_...`.

/data2/mingyu/composed_image_retrieval/eval_all_datasets.sh for eval cirr test split, --mode cirr for cirr validation spilt.

## Experiment Rules
- Never use a launch path that can hide or remap the real target GPU. Always verify experiments by the actual physical GPU occupancy and the first real train/eval log line.
- Treat each dataset as its own tuned setup. Training prompt and evaluation prompt must match for that dataset.
- `CIRR`: use `and` for both training and evaluation.
- `CIRCO`: use `that` for both training and evaluation.
- `FashionIQ`: use `that` for both training and evaluation.
