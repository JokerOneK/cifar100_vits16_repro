import os
import math
import time
import gzip
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torchvision import datasets, transforms
from torch.utils.checkpoint import checkpoint

from tqdm import tqdm

# --- External Libraries Check ---
try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: 'mamba_ssm' library not found. Please install it to use Mamba models.")

try:
    from peft import get_peft_model, LoraConfig, AdaLoraConfig, prepare_model_for_kbit_training

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: 'peft' library not found.")

try:
    import bitsandbytes as bnb

    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

# -------------------------------
# Configuration & Defaults
# -------------------------------
EPOCHS = 10
STEPS_PER_EPOCH = 782
NUM_WORKERS = 4
SEED = 42
MEMORY_CAPACITY_GB = 2.0
DEFAULT_OUTDIR = Path('./mamba_pure_results/tmp')
DEFAULT_DATA_ROOT = Path('./data')

# Set env for better memory handling
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"


def iso_now():
    return datetime.now().isoformat(timespec="seconds")


def set_seed(seed=SEED):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def bytes_to_mib(x: int) -> float:
    return x / (1024.0 * 1024.0)


def ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected.")
    return torch.device("cuda")


# -------------------------------
# Vision Mamba (Vim) Implementation
# -------------------------------

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


# BiMamba removed


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


class BiMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path

        # in_proj and out_proj seem to have no bias in the checkpoint
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize steps
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
            nn.init.constant_(self.dt_proj_b.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_proj_b.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj_b.bias.copy_(inv_dt)

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.A_b_log = nn.Parameter(
            torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D_b = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        batch, seqlen, dim = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, L, D_inner)

        x = x.transpose(1, 2)  # (B, D_inner, L)

        # Forward branch
        x_f = self.conv1d(x)[:, :, :seqlen]
        x_f = nn.functional.silu(x_f)
        y_f = self.ssm(x_f, self.x_proj, self.dt_proj, self.A_log, self.D)

        # Backward branch - flip sequence
        x_b = x.flip([-1])
        x_b = self.conv1d_b(x_b)[:, :, :seqlen]
        x_b = nn.functional.silu(x_b)
        y_b = self.ssm(x_b, self.x_proj_b, self.dt_proj_b, self.A_b_log, self.D_b)

        # Sum forward and flipped-backward
        y = y_f + y_b.flip([-1])

        z = nn.functional.silu(z.transpose(1, 2))
        y = y * z

        out = self.out_proj(y.transpose(1, 2))
        return out

    def ssm(self, u, x_proj, dt_proj, A_log, D):
        # u: (B, D_inner, L)
        L = u.size(-1)
        # x_proj takes (B, L, D_inner) -> needs transpose u
        x_dbl = x_proj(u.transpose(1, 2))  # (B, L, dt_rank + 2*d_state)
        d_dt_rank = self.dt_rank
        d_state = self.d_state

        dt, B, C = torch.split(x_dbl, [d_dt_rank, d_state, d_state], dim=-1)
        dt = dt_proj(dt).transpose(1, 2)  # (B, D_inner, L)
        B = B.transpose(1, 2)
        C = C.transpose(1, 2)

        A = -torch.exp(A_log)

        if self.use_fast_path and selective_scan_fn is not None:
            y = selective_scan_fn(
                u, dt, A, B, C, D.float(), z=None, delta_bias=dt_proj.bias.float(), delta_softplus=True
            )
        else:
            # Fallback or simple implementation if needed, but we expect fast path
            # Minimal placeholder if fast path fails would be complex to write here
            y = selective_scan_fn(
                u, dt, A, B, C, D.float(), z=None, delta_bias=dt_proj.bias.float(), delta_softplus=True
            )
        return y


class VimBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, bidirectional=False):
        super().__init__()
        # Pretrained Vim weights use LayerNorm with NO BIAS
        # However, keep standard epsilon. Checkpoints usually have eps=1e-5 or 1e-6.
        # Vim-Small usually default to 1e-6.
        self.norm = nn.LayerNorm(dim, eps=1e-6, bias=False)

        if bidirectional:
            self.mixer = BiMamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        elif MAMBA_AVAILABLE:
            # Fallback to standard Mamba for pure training from scratch if requested
            # configuration for standard Mamba in mamba_ssm
            self.mixer = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mixer = nn.Identity()

    def forward(self, x):
        x = x + self.mixer(self.norm(x))
        return x


class VisionMamba(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 depth=24,
                 embed_dim=384,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 num_classes=100,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 mid_cls_token=False,
                 bidirectional=False):
        super().__init__()

        self.mid_cls_token = mid_cls_token
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            VimBlock(dim=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand, bidirectional=bidirectional)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6, bias=False)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, N, C = x.shape

        if self.mid_cls_token:
            # Mid-Cls strategy: Add CLS token in middle (N // 2)
            cls_token = self.cls_token.expand(B, -1, -1)
            mid_idx = N // 2
            x = torch.cat((x[:, :mid_idx, :], cls_token, x[:, mid_idx:, :]), dim=1)
        else:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.mid_cls_token:
            num_patches = self.patch_embed.num_patches
            mid_idx = num_patches // 2
            x = x[:, mid_idx]
        else:
            x = x[:, 0]

        x = self.head(x)
        return x


# -------------------------------
# Data Loaders
# -------------------------------
def make_dataloaders_dtd(steps_per_epoch: int, batch_size: int, num_workers: int, data_root: str, download: bool,
                         eval_batch_size: int = 64):
    img_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    dtd_train = datasets.DTD(root=str(root), split="train", transform=train_tf, download=download)
    dtd_val = datasets.DTD(root=str(root), split="val", transform=train_tf, download=download)
    dtd_test = datasets.DTD(root=str(root), split="test", transform=test_tf, download=download)

    num_classes = len(dtd_train.classes)

    train_ds = ConcatDataset([dtd_train, dtd_val])
    test_ds = dtd_test

    num_samples = steps_per_epoch * batch_size
    train_sampler = RandomSampler(train_ds, replacement=True, num_samples=num_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader, num_classes


def make_dataloaders_cifar100(steps_per_epoch: int, batch_size: int, num_workers: int, data_root: str, download: bool,
                              eval_batch_size: int = 64):
    img_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.CIFAR100(root=str(root), train=True, transform=train_tf, download=download)
    test_ds = datasets.CIFAR100(root=str(root), train=False, transform=test_tf, download=download)
    num_classes = 100

    num_samples = steps_per_epoch * batch_size
    train_sampler = RandomSampler(train_ds, replacement=True, num_samples=num_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader, num_classes


# -------------------------------
# Logging Utils (Enhanced)
# -------------------------------

def safe_log10(x: float) -> float:
    if x is None or math.isnan(x) or x <= 0:
        return float("nan")
    return math.log10(x)


def compute_sam(acc_pct: float, energy_j: float, ab_values: List[int]) -> Dict[str, float]:
    acc = acc_pct / 100.0
    results = {}
    logE = safe_log10(energy_j)
    for a in ab_values:
        key = f"SAM_a{a}_b{a}"
        if math.isnan(logE) or acc <= 0:
            results[key] = float("nan")
        else:
            results[key] = (acc ** a) / (logE ** a)
    return results


def ensure_metrics_csv_header(ab_values: List[int], metrics_path: Path):
    first = (not metrics_path.exists()) or (metrics_path.stat().st_size == 0)
    if first:
        with open(metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            header = ["epoch", "train_time_s", "eval_time_s", "total_time_s", "train_energy_j", "eval_energy_j",
                      "total_energy_j", "avg_power_w", "test_acc_pct"]
            for a in ab_values:
                header.append(f"SAM_a{a}_b{a}")
            w.writerow(header)


class GpuPowerMeter:
    """ Enhanced Power Meter from MambaVision script """

    def __init__(self, device_index: int, step_energy_path: Path):
        self.available = False
        self.handle = None
        self.device_index = device_index
        self._init_nvml()

        self.reset_epoch()
        self._step_file = gzip.open(step_energy_path, "at", newline="")
        self._step_writer = csv.writer(self._step_file)
        if step_energy_path.stat().st_size == 0:
            self._step_writer.writerow(
                ["ts", "epoch", "step", "phase", "step_ms", "p_start_w", "p_end_w", "p_avg_w", "energy_j"])

    def _init_nvml(self):
        try:
            import pynvml
            self.nvml = pynvml
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(self.device_index)
            _ = self.nvml.nvmlDeviceGetPowerUsage(self.handle)
            self.available = True
        except Exception:
            self.available = False
            self.nvml = None
            self.handle = None

    def close(self):
        try:
            if self.available and self.nvml:
                self.nvml.nvmlShutdown()
        except Exception:
            pass
        try:
            self._step_file.close()
        except Exception:
            pass

    def sample_power_w(self) -> float:
        if not self.available: return float("nan")
        try:
            return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            return float("nan")

    def reset_epoch(self):
        self.train_energy_j = 0.0
        self.eval_energy_j = 0.0
        self.train_time_s = 0.0
        self.eval_time_s = 0.0

    def _accumulate(self, phase: str, step_time_s: float, p_start: float, p_end: float):
        p_avg = (p_start + p_end) / 2.0 if (not math.isnan(p_start) and not math.isnan(p_end)) else float("nan")
        e = p_avg * step_time_s if not math.isnan(p_avg) else float("nan")

        if phase == "train" or phase.startswith("train"):
            self.train_time_s += step_time_s
            if math.isnan(p_avg):
                self.train_energy_j = float("nan")
            elif not math.isnan(self.train_energy_j):
                self.train_energy_j += e
        else:
            self.eval_time_s += step_time_s
            if math.isnan(p_avg):
                self.eval_energy_j = float("nan")
            elif not math.isnan(self.eval_energy_j):
                self.eval_energy_j += e
        return p_avg, e

    def log_step(self, phase: str, epoch: int, step: int, step_time_s: float, p_start: float, p_end: float):
        p_avg, e = self._accumulate(phase, step_time_s, p_start, p_end)
        self._step_writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            epoch, step, phase, f"{step_time_s * 1000:.3f}",
            f"{p_start:.3f}", f"{p_end:.3f}",
            f"{p_avg:.3f}" if not math.isnan(p_avg) else "nan",
            f"{e:.6f}" if not math.isnan(e) else "nan"
        ])

    def epoch_totals(self):
        total_e = (self.train_energy_j if not math.isnan(self.train_energy_j) else 0.0) + \
                  (self.eval_energy_j if not math.isnan(self.eval_energy_j) else 0.0)
        total_e = total_e if (not math.isnan(self.train_energy_j) or not math.isnan(self.eval_energy_j)) else float(
            "nan")
        total_t = self.train_time_s + self.eval_time_s
        avg_power = (total_e / total_t) if (not math.isnan(total_e) and total_t > 0) else float("nan")
        return dict(
            train_energy_j=self.train_energy_j, eval_energy_j=self.eval_energy_j,
            total_energy_j=total_e, train_time_s=self.train_time_s, eval_time_s=self.eval_time_s,
            total_time_s=total_t, avg_power_w=avg_power
        )


# -------------------------------
# Pretrained Weight Loading
# -------------------------------
def load_pretrained_weights(model, model_name="vim_small_midclstok"):
    """
    Load official weights from Hugging Face for Vim-Small-MidCls.
    """
    if "midclstok" in model_name:
        url = "https://huggingface.co/hustvl/Vim-small-midclstok/resolve/main/vim_s_midclstok_80p5acc.pth"
    else:
        print(f"Warning: Unknown model name {model_name}, skipping pretrained load.")
        return model

    print(f"Downloading/Loading pretrained weights from {url}...")
    try:
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)
    except Exception as e:
        print(f"Error loading from URL: {e}. Trying to load local file if exists...")
        local_path = Path("./vim_s_midclstok_80p5acc.pth")
        if local_path.exists():
            checkpoint = torch.load(local_path, map_location="cpu")
        else:
            raise RuntimeError(f"Could not load weights. Error: {e}")

    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Remap keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("layers."):
            new_k = k.replace("layers.", "blocks.")
        elif k.startswith("norm_f."):
            new_k = k.replace("norm_f.", "norm.")
        else:
            new_k = k
        new_state_dict[new_k] = v

    # Filter out head if num_classes doesn't match
    if model.head.weight.shape[0] != new_state_dict.get("head.weight", torch.empty(0)).shape[0]:
        print(
            f"Head mismatch (Pretrained: {new_state_dict.get('head.weight', torch.tensor([])).shape}, Current: {model.head.weight.shape}). Dropping head weights.")
        if "head.weight" in new_state_dict: del new_state_dict["head.weight"]
        if "head.bias" in new_state_dict: del new_state_dict["head.bias"]

    # Load with strict=False to allow for missing head, but check for crucial mismatches
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded weights with msg: {msg}")
    return model


# -------------------------------
# Training Logic per Run
# -------------------------------
# -------------------------------
# Training Logic per Run
# -------------------------------
def train_one_run(dataset_name: str, batch_size: int, peft_method: str, args, device: torch.device, base_outdir: Path):
    run_id = f"{dataset_name}_bs{batch_size}_{peft_method}"
    outdir = base_outdir / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = outdir / "metrics.csv"
    power_path = outdir / "power.csv.gz"

    # SAM parameters
    ab_values = [1, 2, 3, 4, 5]
    ensure_metrics_csv_header(ab_values, metrics_path)

    print(f"\n{'=' * 40}")
    print(f"STARTING RUN: {run_id}")
    print(f"Dataset: {dataset_name} | Batch Size: {batch_size} | Mode: {peft_method}")
    print(f"{'=' * 40}\n")

    # 1. Data
    print(f"Loading {dataset_name}...")
    if dataset_name == "dtd":
        train_loader, val_loader, num_classes = make_dataloaders_dtd(
            STEPS_PER_EPOCH, batch_size, NUM_WORKERS, args.data_root, True
        )
    else:
        train_loader, val_loader, num_classes = make_dataloaders_cifar100(
            STEPS_PER_EPOCH, batch_size, NUM_WORKERS, args.data_root, True
        )

    # 2. Model
    print("Creating Vision Mamba (Vim-Small equivalent)...")
    mid_cls_token = False
    bidirectional = False
    if args.pretrained:
        mid_cls_token = True  # Official Vim-Small-MidCls uses this strategy
        bidirectional = True
        print("Using Mid-Cls-Token & BiMamba Strategy for Pretrained Weights")

    model = VisionMamba(
        img_size=224,
        patch_size=16,
        depth=24,
        embed_dim=384,
        d_state=16,
        d_conv=4,
        expand=2,
        num_classes=num_classes,
        mid_cls_token=mid_cls_token,
        bidirectional=bidirectional
    )

    if args.pretrained:
        model = load_pretrained_weights(model, "vim_small_midclstok")

    # 3. PEFT & LR
    if peft_method == "none":
        lr = 1e-4
        print(f"[Mode: None] Using lower Learning Rate: {lr}")
    else:
        lr = 1e-3
        print(f"[Mode: {peft_method}] Using standard PEFT Learning Rate: {lr}")

    if peft_method == "qlora" and BNB_AVAILABLE:
        print(
            "Applying QLoRA (4-bit placeholder/quantization not fully implemented here, using standard LoRA + load checks if needed)...")
        # In a real QLoRA setup, we would load the model in 4bit using BitsAndBytesConfig
        # Here we just treat it as LoRA logic for simplicity unless user provided 4bit loaded model
        pass

    target_modules = ["in_proj", "x_proj", "dt_proj", "out_proj", "proj"]

    if peft_method in ["lora", "qlora"]:
        print(f"Applying LoRA (r={args.lora_r})...")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["head"],
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    elif peft_method == "adalora":
        print(f"Applying AdaLoRA (r={args.lora_r})...")
        total_steps = args.epochs * STEPS_PER_EPOCH
        # AdaLoRA does not support Conv2d (proj), so we exclude it
        adalora_targets = ["in_proj", "x_proj", "dt_proj", "out_proj"]
        config = AdaLoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=adalora_targets,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["head"],
            # AdaLoRA specific
            init_r=12,
            tinit=200,
            tfinal=1000,
            deltaT=10,
            total_step=total_steps
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    elif peft_method == "bitfit":
        print("Applying BitFit (Bias-Tuning)...")
        # Freeze all
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Unfreeze bias and head
        trainable_count = 0
        total_count = 0
        for name, param in model.named_parameters():
            if "bias" in name or "head" in name:
                param.requires_grad = True
                trainable_count += param.numel()
            total_count += param.numel()

        print(
            f"BitFit: Trainable params: {trainable_count / 1e6:.2f}M / {total_count / 1e6:.2f}M ({trainable_count / total_count * 100:.2f}%)")

    else:
        if peft_method != "none":
            print(f"Warning: Unknown PEFT method {peft_method}, defaulting to full fine-tuning.")
        print(f"Full Fine-Tuning / Training From Scratch")
        print(f"Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    pwr = GpuPowerMeter(0, power_path)

    best_acc = 0.0

    for epoch in range(args.epochs):
        pwr.reset_epoch()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"[{run_id}] Ep {epoch + 1}/{args.epochs}")
        for step, (x, y) in enumerate(pbar):
            torch.cuda.synchronize()
            p_start = pwr.sample_power_w()
            t0 = time.time()

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            # Record stats
            torch.cuda.synchronize()
            step_t = time.time() - t0
            p_end = pwr.sample_power_w()
            pwr.log_step("train", epoch + 1, step, step_t, p_start, p_end)

            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.3f}"})

            if args.dry_run:
                break

        if not args.dry_run:
            scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        # Eval loop with power logging too
        with torch.no_grad():
            for step, (x, y) in enumerate(val_loader):
                torch.cuda.synchronize()
                p_start = pwr.sample_power_w()
                t0 = time.time()

                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

                torch.cuda.synchronize()
                step_t = time.time() - t0
                p_end = pwr.sample_power_w()
                pwr.log_step("eval", epoch + 1, step, step_t, p_start, p_end)

                if args.dry_run: break

        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch + 1} Val Acc: {val_acc:.4f}")

        # Metrics logging
        totals = pwr.epoch_totals()
        sam_res = compute_sam(val_acc * 100, totals['total_energy_j'], ab_values)

        with open(metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            # ["epoch","train_time_s","eval_time_s","total_time_s","train_energy_j","eval_energy_j","total_energy_j","avg_power_w","test_acc_pct"] + SAMs
            row = [
                epoch + 1,
                f"{totals['train_time_s']:.3f}",
                f"{totals['eval_time_s']:.3f}",
                f"{totals['total_time_s']:.3f}",
                f"{totals['train_energy_j']:.3f}",
                f"{totals['eval_energy_j']:.3f}",
                f"{totals['total_energy_j']:.3f}",
                f"{totals['avg_power_w']:.3f}",
                f"{val_acc * 100:.2f}"
            ]
            for a in ab_values:
                key = f"SAM_a{a}_b{a}"
                val = sam_res.get(key, float("nan"))
                row.append(f"{val:.4e}")
            w.writerow(row)

        # Save Best & Last
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), outdir / "best_model.pth")
            with open(outdir / "best_acc.txt", "w") as f:
                f.write(f"{best_acc:.4f}")

        torch.save(model.state_dict(), outdir / f"last_model.pth")

        if args.dry_run:
            print("Dry run finished.")
            break

    pwr.close()
    print(f"Finished Run: {run_id} | Best Acc: {best_acc:.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs='+', default=["cifar100"], help="List of datasets")
    parser.add_argument("--batch_sizes", type=int, nargs='+', default=[16], help="List of batch sizes")

    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)

    # Updated choices including 'all', 'adalora', 'bitfit'
    parser.add_argument("--peft", type=str, choices=["none", "lora", "qlora", "adalora", "bitfit", "all"],
                        default="none")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true", help="Run 1 step to verify")
    parser.add_argument("--pretrained", action="store_true", help="Load official Vim-Small-MidCls ImageNet weights")

    args = parser.parse_args()

    set_seed(SEED)
    device = ensure_cuda()
    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    # Determine PEFT methods list
    if args.peft == "all":
        peft_methods = ["none", "lora", "qlora", "adalora", "bitfit"]
    else:
        peft_methods = [args.peft]

    print(f"Planned Runs: Datasets={args.datasets}, BatchSizes={args.batch_sizes}, PEFT={peft_methods}")

    for dataset in args.datasets:
        for batch_size in args.batch_sizes:
            for peft_method in peft_methods:
                try:
                    train_one_run(dataset, batch_size, peft_method, args, device, base_outdir)
                except Exception as e:
                    print(f"Error in run: {e}")


if __name__ == "__main__":
    main()
