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

import os
import time
import logging
from time import gmtime, strftime
from pathlib import Path
import json
from functools import partial
import wandb
import torch
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as T
from PIL import Image
import math
import torch.nn as nn
import torch.nn.functional as F

from model.clip import _transform, load
from model.model import convert_weights, CLIP, IM2TEXT, enable_lora_on_clip
from eval_utils import evaluate_imgnet_retrieval, evaluate_coco, evaluate_fashion, evaluate_cirr, evaluate_cirr_test
from data import CsvDataset, CustomFolder, ImageList, CsvCOCO, FashionIQ, CIRR
from params import parse_args, get_project_root
from logger import setup_primary_logging, setup_worker_logging
from utils import is_master, convert_models_to_fp32, TargetPad


def _safe_torch_load(path, map_location=None, weights_only=False):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only, mmap=True)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=weights_only)


# -------------------
# LoRA (minimal, local) - 与当前 main.py 保持一致
# -------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 64, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.empty(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        out = self.base(x)
        x_d = self.dropout(x)
        A = self.A.to(dtype=x_d.dtype)
        B = self.B.to(dtype=x_d.dtype)
        lora = (x_d @ A.t()) @ B.t()
        return out + (self.scaling * lora).to(dtype=out.dtype)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


def apply_lora_to_linear_layers(module: nn.Module, r: int, alpha: int, dropout: float = 0.0):
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            apply_lora_to_linear_layers(child, r=r, alpha=alpha, dropout=dropout)

def load_model(args):
    model, _, preprocess_val = load(
            args.model,
            jit=False)
    img2text = IM2TEXT(embed_dim=model.embed_dim, 
                       middle_dim=args.middle_dim, 
                       output_dim=model.token_embedding.weight.shape[1],
                       n_layer=args.n_layer) 
    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32" or args.gpu is None:
        convert_models_to_fp32(model)

    # ---- Enable LoRA on CLIP (必须与训练时一致) ----
    # 注意：LoRA应该在加载权重之前应用，这样模型结构才能匹配checkpoint。
    # 使用 q/k/v-aware 的 LoRA 包装，才能接住 attention-only LoRA key。
    if not getattr(args, "no_lora", False):
        enable_lora_on_clip(
            model,
            r=getattr(args, "lora_r", 64),
            alpha=getattr(args, "lora_alpha", 16),
            dropout=getattr(args, "lora_dropout", 0.0),
        )

    if not torch.cuda.is_available():
        model.float()
        img2text.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        img2text.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
        # Previously batch size and workers were global and not per GPU.
        # args.batch_size = args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, 
                device_ids=[args.gpu], 
                find_unused_parameters=model.has_extra)
            img2text = torch.nn.parallel.DistributedDataParallel(img2text, 
                device_ids=[args.gpu], find_unused_parameters=False)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
            img2text = torch.nn.DataParallel(img2text, device_ids=args.multigpu)

        if args.precision == "fp16":
            convert_weights(model)
            convert_weights(img2text)
    if args.resume == 'auto':
        checkpoint_list = os.listdir(args.checkpoint_path)
        checkpoint_list = [ckpt for ckpt in checkpoint_list if ckpt.startswith('epoch')]
        if checkpoint_list:
            latest_epoch = max([int(ckpt.split('_')[1].split('.')[0]) for ckpt in checkpoint_list])
            args.resume = os.path.join(args.checkpoint_path, f'epoch_{latest_epoch}.pt')
        else:
            args.resume = None

    assert args.resume is not None
    if os.path.isfile(args.resume):
        if args.gpu is None:
            checkpoint = _safe_torch_load(args.resume)
        else:
            # Load to CPU first to avoid OOM from optimizer states in large checkpoints.
            checkpoint = _safe_torch_load(args.resume, map_location="cpu", weights_only=False)
            print("Checkpoint top-level keys:", checkpoint.keys())
            for k in checkpoint.keys():
                if "state_dict" in k:
                    value = checkpoint[k]
                    if value is None:
                        print(k, "is None")
                    else:
                        print(k, "num params =", len(value))

        sd = checkpoint["state_dict"]
        sd_img2text = checkpoint["state_dict_img2text"]
        if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        if not args.distributed and next(iter(sd_img2text.items()))[0].startswith('module'):
            sd_img2text = {k[len('module.'):]: v for k, v in sd_img2text.items()}

                # ===== debug: check resume coverage =====
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing_i2t, unexpected_i2t = img2text.load_state_dict(sd_img2text, strict=False)

        print("\n==== Resume load debug ====")
        print("CLIP missing keys:", len(missing))
        print("CLIP unexpected keys:", len(unexpected))
        print("IM2TEXT missing keys:", len(missing_i2t))
        print("IM2TEXT unexpected keys:", len(unexpected_i2t))

        # 只看看前几十个，避免刷屏
        for k in missing[:30]:
            print("  CLIP MISSING:", k)
        for k in unexpected[:30]:
            print("  CLIP UNEXPECTED:", k)
        print("==== End debug ====\n")
        # ===== debug end =====

        # 使用 strict=False 加载以兼容 LoRA 和非 LoRA 权重
        model.load_state_dict(sd, strict=False)
        img2text.load_state_dict(sd_img2text, strict=False)
        logging.info(
            f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
        )
    else:
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
    return model, img2text, preprocess_val

def setup_log_save(args):
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"{name}: {val}")
                f.write(f"{name}: {val}\n")
            
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    if args.dp:
        args.batch_size *= args.world_size
    if args.gpu is not None:
        logging.info(f"Use GPU: {args.gpu} for training")
        torch.cuda.set_device(args.gpu)


def main_worker(gpu, ngpus_per_node, log_queue, args):
    args.gpu = gpu
    args.rank = gpu
    setup_worker_logging(args.rank, log_queue, args.log_level)
    # Log and save params.
    setup_log_save(args)
    # Load trained model
    model, img2text, preprocess_val = load_model(args)
    cudnn.benchmark = True
    cudnn.deterministic = False

    # Override logit_scale if specified
    if getattr(args, 'logit_scale', None) is not None:
        m = model.module if args.distributed or args.dp else model
        m.logit_scale.data = torch.log(torch.tensor(args.logit_scale)).to(m.logit_scale.device)
        logging.info(f"Overriding logit_scale to exp({args.logit_scale:.4f}) = logit_scale.data = {m.logit_scale.data.item():.4f}")   
    root_project = os.path.join(get_project_root(), 'data')
    ## Padding option
    if args.target_pad:
        trans_tmp = preprocess_val.transforms
        trans_tmp = [TargetPad(1.25)] + trans_tmp
        preprocess_train = T.Compose(trans_tmp)
        preprocess_val = preprocess_train

     ## Load data for each evaluation dataset and perform evaluation.
    if args.eval_mode == 'coco':
        trans_val = preprocess_val.transforms
        n_px = trans_val[1].size
        trans_val = [T.Resize(n_px, interpolation=Image.BICUBIC)] + trans_val[2:]
        preprocess_val_region = T.Compose(trans_val)
        source_dataset = CsvCOCO(transforms=preprocess_val, 
                                 transforms_region=preprocess_val_region, 
                                 root=root_project)
        source_dataloader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
        evaluate_coco(model, img2text, args, source_dataloader)

    elif args.eval_mode == 'cirr':
        source_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project)
        target_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project, 
                              mode='imgs')
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        print("num queries:", len(source_dataset))
        print("num gallery:", len(target_dataset))
        evaluate_cirr(model, 
                      img2text, 
                      args, 
                      source_dataloader, 
                      target_dataloader)

    elif args.eval_mode == 'cirr_test':
        source_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project, test=True)
        target_dataset = CIRR(transforms=preprocess_val, 
                              root=root_project, 
                              mode='imgs', 
                              test=True)
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        results = evaluate_cirr_test(model,
                                     img2text,
                                     args,
                                     source_dataloader,
                                     target_dataloader)
        output_dir = getattr(args, 'cirr_output_dir', 'res_cirr')
        os.makedirs(output_dir, exist_ok=True)
        for feat_key, feat_results in results.items():
            # feat_results = {"recall": {...}, "recall_subset": {...}}
            recall_dict = feat_results.get("recall", feat_results)
            subset_dict = feat_results.get("recall_subset", None)
            with open(os.path.join(output_dir, feat_key + '.json'), 'w') as f:
                json.dump(recall_dict, f, sort_keys=True)
            if subset_dict and len(subset_dict) > 2:
                with open(os.path.join(output_dir, 'subset_' + feat_key + '.json'), 'w') as f:
                    json.dump(subset_dict, f, sort_keys=True)
        logging.info(f"CIRR test results (recall + subset) saved to {output_dir}/")
    
    elif args.eval_mode == 'fashion':
        assert args.source_data in ['dress', 'shirt', 'toptee']
        source_dataset = FashionIQ(cloth=args.source_data, 
                                   transforms=preprocess_val, 
                                   root=root_project, 
                                   is_return_target_path=True,
                                   image_root=args.fashioniq_image_root,
                                   image_ext=args.fashioniq_image_ext)
        target_dataset = FashionIQ(cloth=args.source_data, 
                                   transforms=preprocess_val, 
                                   root=root_project, 
                                   mode='imgs',
                                   image_root=args.fashioniq_image_root,
                                   image_ext=args.fashioniq_image_ext)
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        evaluate_fashion(model, img2text, args, source_dataloader, target_dataloader)
    elif args.eval_mode == 'imgnet':
        domains = ['cartoon', 'origami', 'toy', 'sculpture']
        prompt = ["a {} of *".format(domain) for domain in domains]
        source_path = os.path.join(root_project, "imgnet", "imgnet_real_query.txt")
        target_path = os.path.join(root_project, "imgnet", "imgnet_targets.txt")
        source_dataset = ImageList(source_path, root=root_project, transforms=preprocess_val, is_labels=True)
        target_dataset = ImageList(target_path, root=root_project, transforms=preprocess_val, is_labels=True)
        eval_func = evaluate_imgnet_retrieval
        source_dataloader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        target_dataloader = DataLoader(
            target_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        eval_func(model, img2text, args, prompt, source_dataloader, target_dataloader)



    elif args.eval_mode == 'circo':
        from data import CIRCODataset, CustomFolder
        from eval_utils import evaluate_circo

        logging.info("Evaluating on CIRCO Dataset (TEST SPLIT)")
        
        # 1. 构建 Gallery Loader (COCO Unlabeled 2017)
        # 路径通常是 ./data/CIRCO/COCO2017_unlabeled/unlabeled2017
        gallery_path = os.path.join(root_project, 'CIRCO', 'COCO2017_unlabeled', 'unlabeled2017')
        
        if not os.path.exists(gallery_path):
            raise FileNotFoundError(f"CIRCO Gallery path not found: {gallery_path}")

        # Gallery 包含所有 COCO Unlabeled 图片
        gallery_dataset = CustomFolder(gallery_path, transform=preprocess_val)
        
        gallery_dataloader = DataLoader(
            gallery_dataset,
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False
        )

        # 2. 构建 Query Loader (CIRCO Test Set)
        # ==========================================
        # 修改点：直接写死 split 为 'test'
        # ==========================================
        # ... 前面的代码 ...
        
        # 2. 构建 Query Loader
        split = 'test'
        
        # 构造 CIRCO 数据的绝对路径
        # 假设 root_project 是 ".../data"，那么 data_path 应该是 ".../data/CIRCO"
        circo_data_path = os.path.join(root_project, 'CIRCO')
        
        query_dataset = CIRCODataset(
            data_path=circo_data_path,   # [修正1] 参数名从 root 改为 data_path
            split=split,
            mode='relative',             # [修正2] 补充必填参数 mode，CIR 任务通常用 'relative'
            transforms=preprocess_val,   # 传入图像预处理函数
            preprocess=preprocess_val    # [修正3] 你的定义里还有一个 preprocess，也传同一个函数防止报错
        )
        
        # ... 后面的代码 ...
        
        query_dataloader = DataLoader(
            query_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False
        )

        # 3. 运行评估
        # 传入 root_project 是为了让 eval 函数能找到保存路径等
        args.root_project = root_project 
        
        # 关键修改：为了确保 evaluate_circo 函数内部知道这是 Test 模式
        # (从而保存 submission.json 而不是尝试计算 mAP)，
        # 我们在这里临时修改 args.eval_mode 的值，或者确保你的 evaluate_circo 逻辑是看 split 的
        args.eval_mode = 'circo_test' 
        
        evaluate_circo(model, img2text, args, query_dataloader, gallery_dataloader)

    elif args.eval_mode == 'genecis':
        from data import GeneCISDataset
        from eval_utils import evaluate_genecis
        
        logging.info(f"Evaluating GeneCIS Task: {args.genecis_task}")
        
        # 根据你提供的信息，图片都在这个目录
        img_root = "/data2/mingyu/composed_image_retrieval/data/genecis/VG_100K"
        # JSON 都在这个目录
        json_root = "/data2/mingyu/genecis/genecis"
        
        # === 关键修改：根据任务类型选择图片目录 ===
        if 'object' in args.genecis_task:
            # Object 任务使用 COCO 图片
            # 请修改为你实际存放 COCO Val2017 图片的路径
            # 例如: /data2/mingyu/coco/val2017
            img_root = "/data2/mingyu/composed_image_retrieval/data/coco/val2017" 
        dataset = GeneCISDataset(
            data_root=img_root,
            json_root=json_root,
            task=args.genecis_task,
            transforms=preprocess_val,
            tokenizer=None 
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        
        evaluate_genecis(model, img2text, args, dataloader)

def main():
    args = parse_args()
        # ---- patch: eval 脚本缺省分布式字段 ----
    if not hasattr(args, "distributed"):
        args.distributed = False
    if not hasattr(args, "dp"):
        args.dp = False
    if not hasattr(args, "world_size"):
        args.world_size = 1
    if not hasattr(args, "rank"):
        args.rank = 0
    # get the name of the experiments
    if args.name is None:
        args.name = (f"lr={args.lr}_"
            "wd={args.wd}_"
            "agg={args.aggregate}_"
            "model={args.model}_"
            "batchsize={args.batch_size}_workers={args.workers}")
        if args.time_suffix:
            args.name += "_date=%Y-%m-%d-%H-%M-%S"
            args.name = strftime(args.name, gmtime())

    if args.copy_codebase:
        import sys, subprocess
        from shutil import copytree, ignore_patterns
        new_code_path = os.path.join(args.logs, args.name, "code")
        if os.path.exists(new_code_path):
            print(
                f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
            )
            return -1
        print(f"Copying codebase to {new_code_path}")
        current_code_path = os.path.realpath(__file__)
        for _ in range(3):
            current_code_path = os.path.dirname(current_code_path)
        copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
        print("Done copying code.")
        os.environ["PYTHONPATH"] = f"{os.environ['PYTHONPATH']}:{os.path.join(new_code_path, 'src')}"
        main_file = os.path.join(new_code_path, "src", "training", "main.py")
        argv = sys.argv
        argv.remove('--copy-codebase')
        argv.extend(['--name', args.name])
        command = [sys.executable] + argv
        print("Executing command:", " ".join(command))
        subprocess.check_call(command)
        return 1

    args.log_path = os.path.join(args.logs, args.name, "out.log")
    if os.path.exists(args.log_path) and args.resume is None:
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    assert args.precision in ['amp', 'fp16', 'fp32']
    #assert args.model in ['RN50', 'RN101', 'RN50x4', 'ViT-B/32'] or os.path.exists(args.model)

    args.ngpus_per_node = torch.cuda.device_count()

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        if dirname:
            os.makedirs(dirname, exist_ok=True)
    

    # Set multiprocessing type to spawn.
    # This is important for logging to work with multiprocessing.
    torch.multiprocessing.set_start_method("spawn")

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    log_queue = setup_primary_logging(args.log_path, args.log_level)
    args.world_size = 1
    main_worker(args.gpu, None, log_queue, args)
    print('evaluation done')


if __name__ == "__main__":
    main()
