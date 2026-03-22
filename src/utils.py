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

import math
from contextlib import contextmanager
import numpy as np
import torch
import torchvision.transforms.functional as F

class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio=1.25):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def is_master(args):
    return (not args.distributed) or args.gpu == 0 or args.dp


def _strip_module_prefix(name):
    if name.startswith("module."):
        return name[len("module.") :]
    return name


class ModuleParamEMA:
    def __init__(self, module, decay):
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}.")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {}
        self.reset(module)

    def _iter_named_params(self, module):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            yield _strip_module_prefix(name), param

    def reset(self, module):
        self.shadow = {}
        with torch.no_grad():
            for name, param in self._iter_named_params(module):
                self.shadow[name] = param.detach().float().clone()
        self.num_updates = 0

    def update(self, module):
        if not self.shadow:
            return
        self.num_updates += 1
        with torch.no_grad():
            for name, param in self._iter_named_params(module):
                param_fp32 = param.detach().float()
                if name not in self.shadow:
                    self.shadow[name] = param_fp32.clone()
                    continue
                self.shadow[name].mul_(self.decay).add_(param_fp32, alpha=1.0 - self.decay)

    def apply_to(self, module):
        backup = {}
        with torch.no_grad():
            for name, param in self._iter_named_params(module):
                shadow = self.shadow.get(name)
                if shadow is None:
                    continue
                backup[name] = param.detach().clone()
                param.copy_(shadow.to(device=param.device, dtype=param.dtype))
        return backup

    def restore(self, module, backup):
        if not backup:
            return
        with torch.no_grad():
            for name, param in self._iter_named_params(module):
                if name in backup:
                    param.copy_(backup[name].to(device=param.device, dtype=param.dtype))

    def state_dict(self):
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: tensor.detach().cpu().clone() for name, tensor in self.shadow.items()},
        }

    def load_state_dict(self, state_dict, module=None):
        if state_dict is None:
            if module is not None and not self.shadow:
                self.reset(module)
            return
        self.decay = float(state_dict.get("decay", self.decay))
        self.num_updates = int(state_dict.get("num_updates", 0))
        raw_shadow = state_dict.get("shadow", {})
        self.shadow = {
            _strip_module_prefix(name): tensor.detach().float().clone()
            for name, tensor in raw_shadow.items()
        }
        if module is not None:
            with torch.no_grad():
                for name, param in self._iter_named_params(module):
                    if name not in self.shadow:
                        self.shadow[name] = param.detach().float().clone()


@contextmanager
def use_ema_weights(module_ema_pairs):
    backups = []
    try:
        for module, ema in module_ema_pairs:
            if module is None or ema is None:
                continue
            backups.append((module, ema, ema.apply_to(module)))
        yield
    finally:
        for module, ema, backup in reversed(backups):
            ema.restore(module, backup)
