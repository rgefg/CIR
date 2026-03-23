# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from pathlib import Path

def get_project_root():
    return Path(__file__).parent.parent

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4", "RN50x64",  "RN50x16", "RN50_flat", "RN50_t1", "RN50_t2", "RN50_t3", "RN50_t4", "RN50_t5", "RN50_t6",
                      "RN50_flat_ft", "RN50_t1_pos_ft", "RN50_t2_pos_ft", "RN50_t1_pos", "RN50_t2_pos",
                      "RN50_flat_large", "RN50_t1_large", "RN50_t2_large",
                      "RN50_a2", "RN50_a2s", "ViT-H-14"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B/32", "ViT-L/14", "ViT-B/16"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--genecis-task",
    type=str,
    choices=['focus_attribute', 'change_attribute', 'focus_object', 'change_object'],
    default='focus_attribute',
    help="Specific GeneCIS task to run")
    parser.add_argument("--no-time-suffix",
        default=True,
        action="store_false",
        help="Whether to append current time in the suffix.",
        dest="time_suffix")
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to csv filewith training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to csv file with validation data",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="list of prompts split with ,",
    )
    parser.add_argument(
        "--retrieval-data",
        type=str,
        default=None,
        help="Path to csv file or folder of retrieval data",
    )
    parser.add_argument(
        "--demo-out",
        type=str,
        default="demo",
        help="Path to the output directory for visualization",
    )
    parser.add_argument(
        "--source-data",
        type=str,
        default=None,
        help="Path to txt file of retrieval data",
    )
    parser.add_argument(
        "--target-data",
        type=str,
        default=None,
        help="Path to txt file of retrieval data",
    )
    parser.add_argument(
        "--target-pad",
        action="store_true",
        default=False,
        help="Padding augmentation proposed by combiner.",
    )
    parser.add_argument(
        "--query_file",
        type=str,
        default=None,
        help="Path to query image file for retrieval visualization",
    )
    parser.add_argument("--eval-mode",
        type=str,
        choices=["coco", "cirr", "cirr_test", "fashion", "imgnet", "circo", "genecis"],
        default="coco",
        help="Evaluate Pacs")
    parser.add_argument("--middle_dim",
        default=512,
        type=int,
        help="Number of hidden units in mapping network.")
    parser.add_argument("--droprate",
        default=0.1,
        type=float,
        help="Dropout rate.")
    parser.add_argument(
        "--n-layer", type=int, default=2, help="Number of layers in im2text"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "inet", "auto", "inet,csv","cc3m_cir_wds", "csv,inet", "directory", "fashion-iq", "cirr", "imgnet_r", "circo"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-type-val",
        choices=["webdataset", "csv", "inet", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument("--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument("--use-debiased-sampler",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument("--use-prefix",
        default=False,
        action="store_true",
        help="Whether to use prefix conditioning in using image classification dataset.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Specify a single GPU to run the code on for debugging."
        "Leave at None to use all available GPUs.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-step-start",
        type=int,
        default=0,
        help="Start global step for interval checkpoint saving. Set <=0 to disable.",
    )
    parser.add_argument(
        "--save-step-end",
        type=int,
        default=0,
        help="End global step (inclusive) for interval checkpoint saving.",
    )
    parser.add_argument(
        "--save-step-interval",
        type=int,
        default=0,
        help="Checkpoint save interval for global steps in [save-step-start, save-step-end]. Set <=0 to disable.",
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--regression-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch size = batch_size * world_size * accum_steps.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Log every N optimizer updates (not micro-batches).",
    )
    parser.add_argument(
        "--log-every-s",
        type=float,
        default=60.0,
        help="Also log if at least this many seconds passed since last log (useful when steps are slow).",
    )
    parser.add_argument(
        "--cirr-val-eval-every",
        type=int,
        default=100,
        help="Run CIRR validation evaluation every N optimizer updates during training. Set 0 to disable.",
    )
    parser.add_argument(
        "--model",
        choices=["RN50", "RN101", "RN50x4",  "RN50x64", "RN50x16", "ViT-B/16", "ViT-B/32", "ViT-L/14", "ViT-H-14", 
                 "RN50_flat", "RN50_t1", "RN50_t2", "RN50_t3", "RN50_t4", "RN50_t5", "RN50_t6",
                 "RN50_flat_ft", "RN50_t1_pos_ft", "RN50_t2_pos_ft", "RN50_t1_pos", "RN50_t2_pos",
                 "RN50_flat_large", "RN50_t1_large", "RN50_t2_large",
                 "RN50_a2", "RN50_a2s"],
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--openai-pretrained",
        default=False,
        action='store_true',
        help="Use the openai pretrained models.",
    )
    parser.add_argument(
        "--pic2word-pretrained",
        type=str,
        default=None,
        help="Path to pic2word pretrained model (e.g., checkpoint/pic2word_model.pt)",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:6100",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--skip-aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--C", type=float, default=3.16, help="inverse regularizer for logistic reg."
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )
    parser.add_argument(
        "--dp",
        default=False,
        action="store_true",
        help="Use DP instead of DDP."
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="In DP, which GPUs to use for multigpu training",
    )

        # ---- CC3M CIR (jsonl + webdataset) ----
    parser.add_argument(
        "--cc3m-cir-jsonl",
        type=str,
        default=None,
        help="Path to cc3m_cir_dataset_full.jsonl (id/instruction/modified_caption).",
    )
    parser.add_argument(
        "--cc3m-cir-reverse-jsonl",
        type=str,
        default=None,
        help="Optional sidecar JSONL that supplies reverse_instruction by id without changing the base retrieval set.",
    )
    parser.add_argument(
        "--wds-shards",
        type=str,
        default=None,
        help="WebDataset shards pattern, e.g. /path/cc3m-train-{0000..3317}.tar or pipe:... or hf://...",
    )
    parser.add_argument(
        "--wds-image-key",
        type=str,
        default="jpg;png;jpeg;webp",
        help="Image extensions in WDS sample, semicolon-separated.",
    )
    parser.add_argument(
        "--wds-text-key",
        type=str,
        default="txt;text;caption",
        help="Text/caption keys in WDS sample, semicolon-separated.",
    )
    parser.add_argument(
        "--wds-shuffle",
        type=int,
        default=20000,
        help="Shuffle buffer size for WDS samples.",
    )
    parser.add_argument(
        "--wds-shardshuffle",
        type=int,
        default=1000,
        help="Shard shuffle size for WDS.",
    )
    parser.add_argument(
        "--wds-resampled",
        action="store_true",
        default=True,
        help="Use resampled WebDataset (good for infinite streaming).",
    )
    parser.add_argument(
        "--wds-epoch-steps",
        type=int,
        default=100000,
        help="Steps per epoch when using streaming dataset (since length is unknown).",
    )
    parser.add_argument(
        "--wds-cache-dir",
        type=str,
        default=None,
        help="Local cache directory for WebDataset shards. If set, remote shards will be downloaded and cached here.",
    )
    parser.add_argument(
        "--wds-deterministic",
        action="store_true",
        default=False,
        help="Use fixed seeds for WebDataset shard/sample resampling and shuffling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Base random seed for training reproducibility and fair geo on/off ablations.",
    )
    parser.add_argument(
        "--deterministic-train",
        action="store_true",
        default=False,
        help="Enable deterministic cuDNN settings for fair ablations.",
    )

    # ---- Lcom training hyperparams (paper defaults) ----
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--amp-init-scale", type=float, default=65536.0,
                        help="GradScaler init scale for the retrieval branch.")
    parser.add_argument("--amp-growth-factor", type=float, default=2.0,
                        help="GradScaler growth factor for the retrieval branch.")
    parser.add_argument("--amp-backoff-factor", type=float, default=0.5,
                        help="GradScaler backoff factor for the retrieval branch.")
    parser.add_argument("--amp-growth-interval", type=int, default=2000,
                        help="GradScaler growth interval for the retrieval branch.")
    parser.add_argument("--retrieval-ema-decay", type=float, default=0.0,
                        help="EMA decay for the retrieval branch (CLIP LoRA/logit_scale + img2text). Set <=0 to disable.")
    parser.add_argument("--reset-logit-scale", action="store_true", default=False,
                        help="Reset logit_scale to standard CLIP value (log(1/0.07) ≈ 2.659) at start of training.")
    parser.add_argument("--freeze-logit-scale", action="store_true", default=False,
                        help="Freeze logit_scale (set requires_grad=False) for debugging. Use with --reset-logit-scale.")
    parser.add_argument("--logit-scale", type=float, default=None,
                        help="Override logit_scale (exp value) during eval. E.g., 13.5 for ~2.6 in log space.")
    parser.add_argument("--logit-scale-clamp-min", type=float, default=None,
                        help="Min value for logit_scale clamp during training (in log space, exp value). E.g., 9.0 for ~2.2 in log space.")
    parser.add_argument("--logit-scale-clamp-max", type=float, default=None,
                        help="Max value for logit_scale clamp during training (in log space, exp value). E.g., 36.6 for ~3.6 in log space.")
    parser.add_argument("--logit-scale-freeze-percent", type=float, default=0.3,
                        help="Percentage of steps to freeze logit_scale (default: 0.3 = first 30%%). After this, logit_scale will be trained with clamping.")
    parser.add_argument("--no-lora", action="store_true", default=False)
    parser.add_argument("--geo-weight", type=float, default=0.0,
                        help="Weight applied to the auxiliary geo branch. Set 0 to disable the geo branch.")
    parser.add_argument("--geo-seed", type=int, default=None,
                        help="Optional dedicated seed for the geo branch RNG stream. Defaults to --seed.")
    parser.add_argument("--geo-lr", type=float, default=None,
                        help="Geo branch learning rate. Defaults to --lr.")
    parser.add_argument("--geo-beta1", type=float, default=None,
                        help="Geo branch Adam beta1. Defaults to --beta1.")
    parser.add_argument("--geo-beta2", type=float, default=None,
                        help="Geo branch Adam beta2. Defaults to --beta2.")
    parser.add_argument("--geo-eps", type=float, default=None,
                        help="Geo branch Adam epsilon. Defaults to --eps.")
    parser.add_argument("--geo-wd", type=float, default=None,
                        help="Geo branch weight decay. Defaults to --wd.")
    parser.add_argument("--geo-warmup", type=int, default=None,
                        help="Geo branch warmup steps. Defaults to --warmup.")
    parser.add_argument("--geo-lora-r", type=int, default=None,
                        help="Geo branch LoRA rank. Defaults to --lora-r.")
    parser.add_argument("--geo-lora-alpha", type=int, default=None,
                        help="Geo branch LoRA alpha. Defaults to --lora-alpha.")
    parser.add_argument("--geo-lora-dropout", type=float, default=None,
                        help="Geo branch LoRA dropout. Defaults to --lora-dropout.")
    parser.add_argument("--geo-amp-init-scale", type=float, default=None,
                        help="Geo branch GradScaler init scale. Defaults to --amp-init-scale.")
    parser.add_argument("--geo-amp-growth-factor", type=float, default=None,
                        help="Geo branch GradScaler growth factor. Defaults to --amp-growth-factor.")
    parser.add_argument("--geo-amp-backoff-factor", type=float, default=None,
                        help="Geo branch GradScaler backoff factor. Defaults to --amp-backoff-factor.")
    parser.add_argument("--geo-amp-growth-interval", type=int, default=None,
                        help="Geo branch GradScaler growth interval. Defaults to --amp-growth-interval.")
    parser.add_argument("--geo-ema-decay", type=float, default=None,
                        help="EMA decay for the geo branch. Defaults to --retrieval-ema-decay.")
    parser.add_argument("--ema-eval", action="store_true", default=False,
                        help="Evaluate CIRR validation with EMA weights when EMA is enabled.")
    parser.add_argument("--ema-save-checkpoints", action="store_true", default=False,
                        help="Also save EMA-only checkpoint variants when EMA is enabled.")
    parser.add_argument("--geo-conflict-projection", action="store_true", default=False,
                        help="Project geo gradients away from conflicting retrieval gradients.")
    parser.add_argument("--geo-reverse-weight", type=float, default=0.25,
                        help="Weight for the soft reverse-instruction consistency term in the geo branch.")
    parser.add_argument("--geo-reverse-margin", type=float, default=0.0,
                        help="Penalize geo forward/reverse cosine values above -margin; 0 only penalizes positive cosine.")
    parser.add_argument("--geo-zero-loss-weight", type=float, default=0.0,
                        help="Optional weight for the strong zero-style regularizer ||z_fwd + z_rev|| in the geo branch.")
    parser.add_argument("--geo-embed-norm-eps", type=float, default=1e-6,
                        help="Epsilon used when normalizing geo text embeddings.")
    parser.add_argument("--geo-delta-norm-eps", type=float, default=1e-4,
                        help="Lower bound used when normalizing the geo delta direction z_tgt - z_src.")
    parser.add_argument("--geo-delta-min-norm", type=float, default=1e-3,
                        help="Skip geo samples whose text delta norm is too small to define a stable direction.")
    parser.add_argument("--geo-sampling-mode", type=str, default="all", choices=["all", "hard", "random"],
                        help="Subset strategy for the geo branch within each retrieval batch.")
    parser.add_argument("--geo-topk", type=int, default=0,
                        help="If > 0 and geo-sampling-mode is not all, keep only the top-k selected geo samples per batch.")
    parser.add_argument("--cirr-val-merge-base", type=str, default=None,
                        help="Full checkpoint used as the retrieval base when doing periodic merged CIRR validation.")
    parser.add_argument("--cirr-val-merge-density", type=float, default=0.9,
                        help="TIES density used for periodic merged CIRR validation.")
    parser.add_argument("--cirr-val-merge-weights", type=float, nargs=2, default=[0.5, 0.5],
                        metavar=("BASE_W", "GEO_W"),
                        help="TIES weights for [retrieval_base, current_geo_lora] during merged CIRR validation.")
    parser.add_argument("--cirr-val-merge-gpu", type=int, default=None,
                        help="Dedicated GPU index for periodic merged CIRR validation. Leave unset to disable in-process scheduling.")
    parser.add_argument("--cirr-val-merge-timeout", type=int, default=1800,
                        help="Timeout in seconds for a single merged CIRR validation job launched by helper scripts.")
    parser.add_argument("--prompt-template", type=str, default="A photo of $ that {instruction}")
    parser.add_argument("--cirr-output-dir", type=str, default="res_cirr",
                        help="Output directory for CIRR test results (default: res_cirr)")
    parser.add_argument(
        "--prompt-placeholder",
        type=str,
        default="*",
        help="Single-token placeholder used in prompts, which will be replaced by image tokens (default: '*').",
    )

    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    geo_fallbacks = {
        "geo_seed": args.seed,
        "geo_lr": args.lr,
        "geo_beta1": args.beta1,
        "geo_beta2": args.beta2,
        "geo_eps": args.eps,
        "geo_wd": args.wd,
        "geo_warmup": args.warmup,
        "geo_lora_r": args.lora_r,
        "geo_lora_alpha": args.lora_alpha,
        "geo_lora_dropout": args.lora_dropout,
        "geo_amp_init_scale": args.amp_init_scale,
        "geo_amp_growth_factor": args.amp_growth_factor,
        "geo_amp_backoff_factor": args.amp_backoff_factor,
        "geo_amp_growth_interval": args.amp_growth_interval,
        "geo_ema_decay": args.retrieval_ema_decay,
    }
    for name, val in geo_fallbacks.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
