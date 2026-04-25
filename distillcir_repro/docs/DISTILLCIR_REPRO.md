# DistillCIR Reproduction Notes

This folder is an isolated copy of the Pic2Word-based code plus a DistillCIR student-training path. It does not modify the parent repository `src/`, `model/`, or shell scripts.

## What Is Implemented

- `Lcom`: composed query `(reference image, instruction)` aligned to `modified_caption`.
- `Lrea`: consequential query `(reference image, modified_caption, learnable prompt tokens)` aligned to LLM reasoning text. Existing `brainstorming` fields are used as the reasoning target when no `reasoning` field is present.
- `Lfea`: student composed embedding projected to the cached LLaVA teacher caption embedding space.
- CLIP ViT-L/14 + Pic2Word projection module, LoRA on CLIP visual/text linear layers, full projection-module training.
- `torchrun` DDP launch for physical GPUs 0-7 without `CUDA_VISIBLE_DEVICES` remapping.

## Required Artifacts

- CC3M triplet jsonl:
  `/data2/mingyu/composed_image_retrieval/data/cc3m_cir_dataset_cleaned_v1mid_v2__merged_with_cc3m_new.retrieval_clean_v2.jsonl`
- CC3M WebDataset shards:
  `/data2/mingyu/composed_image_retrieval/data/wds_cache/cc3m-train-{0000..0575}.tar`
- Pic2Word checkpoint:
  `/data2/mingyu/composed_image_retrieval/checkpoint/pic2word_model.pt`
- Teacher embedding cache:
  `/data2/mingyu/composed_image_retrieval/checkpoint/distillcir_teacher/llava_phi3_mini_cc3m_cache`

## Teacher Preparation

1. Download the LLaVA-Phi3 teacher backbone:

```bash
bash scripts/prepare_teacher_backbone.sh
```

2. Train the LLaVA-Phi3 teacher with the same CC3M triplets using contrastive `Lcom`:

```bash
bash scripts/train_teacher_llava_phi3_lora_8x3090.sh
```

This writes a LoRA/projector teacher under:
`/data2/mingyu/composed_image_retrieval/checkpoint/distillcir_teacher/llava_phi3_mini_lora_lcom/final`

3. Cache modified-caption embeddings from that trained teacher:

```bash
TEACHER_ADAPTER=/data2/mingyu/composed_image_retrieval/checkpoint/distillcir_teacher/llava_phi3_mini_lora_lcom/final \
bash scripts/cache_teacher_embeddings_8x3090.sh
```

The cache script writes rank-local shards first, then merges them into `ids.txt`, `embeddings.npy`, and `meta.json`.

## Student Training

CIRR setup uses `and`:

```bash
bash train_with_dropout.sh
```

CIRCO/FashionIQ setup uses `that`:

```bash
bash scripts/train_student_that_8x3090.sh
```

Default global batch is `24 * 8 * 4 = 768`, matching the paper's reported batch size while keeping per-GPU memory conservative for 24GB 3090s.
