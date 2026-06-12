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
import traceback
import gc

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

# Monkey-patch AdaLoRA to handle Mamba SSM Triton kernels that do not populate
# .grad for lora_E parameters (selective-scan custom autograd).  When grad is
# None we treat the parameter's importance as zero, which is semantically
# correct: parameters with no gradient carry no useful information for rank
# allocation.
if PEFT_AVAILABLE:
    try:
        from peft.tuners.adalora.layer import RankAllocator as _RankAllocator

        def _patched_update_ipt(self, model):
            for n, p in model.named_parameters():
                if "lora_" in n and self.adapter_name in n:
                    if n not in self.ipt:
                        self.ipt[n] = torch.zeros_like(p)
                        self.exp_avg_ipt[n] = torch.zeros_like(p)
                        self.exp_avg_unc[n] = torch.zeros_like(p)
                    with torch.no_grad():
                        grad = p.grad if p.grad is not None else torch.zeros_like(p)
                        self.ipt[n] = (p * grad).abs().detach()
                        self.exp_avg_ipt[n] = (
                            self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                        )
                        self.exp_avg_unc[n] = (
                            self.beta2 * self.exp_avg_unc[n]
                            + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                        )

        _RankAllocator.update_ipt = _patched_update_ipt
        print("[AdaLoRA] Patched RankAllocator.update_ipt for Mamba SSM compatibility.")
    except Exception as _e:
        print(f"[AdaLoRA] Monkey-patch failed (non-fatal): {_e}")

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
DEFAULT_OUTDIR = Path('./mamba_pure_results')
DEFAULT_DATA_ROOT = Path('./data')


# Set env for better memory handling
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")

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

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (matches mamba_ssm's RMSNorm)"""
    def __init__(self, d_model: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * norm).to(dtype) * self.weight


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

        # Average forward and flipped-backward (if_divide_out=True, matches official Vim)
        y = (y_f + y_b.flip([-1])) / 2.0

        z = nn.functional.silu(z.transpose(1, 2))
        y = y * z

        out = self.out_proj(y.transpose(1, 2))
        return out

    def ssm(self, u, x_proj, dt_proj, A_log, D):
        # u: (B, D_inner, L)
        L = u.size(-1)
        x_dbl = x_proj(u.transpose(1, 2))  # (B, L, dt_rank + 2*d_state)
        d_dt_rank = self.dt_rank
        d_state = self.d_state

        dt, B_ssm, C_ssm = torch.split(x_dbl, [d_dt_rank, d_state, d_state], dim=-1)
        # Apply dt_proj weight only (no bias) — bias is handled separately
        dt = nn.functional.linear(dt, dt_proj.weight).transpose(1, 2)  # (B, D_inner, L)
        B_ssm = B_ssm.transpose(1, 2)     # (B, d_state, L)
        C_ssm = C_ssm.transpose(1, 2)     # (B, d_state, L)

        A = -torch.exp(A_log)  # (D_inner, d_state)

        if self.use_fast_path and selective_scan_fn is not None:
            try:
                y = selective_scan_fn(
                    u, dt, A, B_ssm, C_ssm, D.float(),
                    z=None, delta_bias=dt_proj.bias.float(), delta_softplus=True
                )
                return y
            except RuntimeError:
                pass  # Fall through to Python fallback

        # Pure-PyTorch SSM fallback (works with autograd, no custom CUDA)
        # Add bias manually then apply softplus (matching selective_scan_fn convention)
        dt = nn.functional.softplus(dt + dt_proj.bias.float().unsqueeze(0).unsqueeze(-1))
        # dt: (B, D_inner, L),  A: (D_inner, d_state)
        batch = u.shape[0]
        d_inner = u.shape[1]
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            dt_i = dt[:, :, i].unsqueeze(-1)          # (B, D_inner, 1)
            B_i = B_ssm[:, :, i].unsqueeze(1)          # (B, 1, d_state)
            C_i = C_ssm[:, :, i].unsqueeze(1)          # (B, 1, d_state)
            u_i = u[:, :, i].unsqueeze(-1)             # (B, D_inner, 1)
            dA = torch.exp(dt_i * A.unsqueeze(0))      # (B, D_inner, d_state)
            dB = dt_i * B_i                             # (B, D_inner, d_state)
            h = h * dA + u_i * dB
            y_i = (h * C_i).sum(dim=-1)                # (B, D_inner)
            y_i = y_i + D * u[:, :, i]
            ys.append(y_i)
        y = torch.stack(ys, dim=-1)  # (B, D_inner, L)
        return y


class VimBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, bidirectional=False, drop_path=0.):
        super().__init__()
        # Official Vim uses RMSNorm with eps=1e-5, no bias
        self.norm = RMSNorm(dim, eps=1e-5)

        if bidirectional:
            self.mixer = BiMamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        elif MAMBA_AVAILABLE:
            self.mixer = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mixer = nn.Identity()

        from timm.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm(x)))
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
                 drop_path_rate=0.0,
                 mid_cls_token=False,
                 bidirectional=False):
        super().__init__()

        self.mid_cls_token = mid_cls_token
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay (matches official Vim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Use 'layers' to match official Vim checkpoint key naming
        self.layers = nn.ModuleList([
            VimBlock(dim=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand,
                     bidirectional=bidirectional, drop_path=dpr[i])
            for i in range(depth)
        ])

        # Use 'norm_f' to match official Vim checkpoint key naming
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
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

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
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

    # No key remapping needed — model attribute names now match official checkpoint:
    #   self.layers (not blocks), self.norm_f (not norm)

    # Filter out head if num_classes doesn't match
    if model.head.weight.shape[0] != state_dict.get("head.weight", torch.empty(0)).shape[0]:
        print(
            f"Head mismatch (Pretrained: {state_dict.get('head.weight', torch.tensor([])).shape}, "
            f"Current: {model.head.weight.shape}). Dropping head weights.")
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

    # Load with strict=False
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights:")
    if msg.missing_keys:
        print(f"  Missing keys: {msg.missing_keys}")
    if msg.unexpected_keys:
        print(f"  Unexpected keys: {msg.unexpected_keys}")
    matched = len(state_dict) - len(msg.unexpected_keys)
    print(f"  Matched: {matched}/{len(state_dict)} checkpoint keys")
    return model

# -------------------------------
# Memory Logger (per-run files) with deferred sync
# -------------------------------
class MemLogger:
    def __init__(self, device, n_layers: int, raw_log_gz: Path, epoch_avg_csv: Path,
                 layer_times_gz: Path, layer_time_epoch_avg_csv: Path):
        self.device = device
        self.n_layers = n_layers
        self.raw_path = raw_log_gz
        self.epoch_avg_path = epoch_avg_csv
        self.layer_times_path = layer_times_gz
        self.layer_time_epoch_avg_path = layer_time_epoch_avg_csv

        self.raw_file = gzip.open(self.raw_path, "at", newline="")
        self.raw_writer = csv.writer(self.raw_file)
        if self.raw_path.stat().st_size == 0:
            self.raw_writer.writerow(["epoch","step","phase","layer","mem_mib"])
        self.epoch_acc: Dict[Tuple[str, int], Tuple[float, int]] = {}

        self.time_file = gzip.open(self.layer_times_path, "at", newline="")
        self.time_writer = csv.writer(self.time_file)
        if self.layer_times_path.stat().st_size == 0:
            self.time_writer.writerow(["ts","epoch","step","phase","layer","ms"])
        self.epoch_time_acc: Dict[Tuple[str, int], Tuple[float, int]] = {}
        self.pending_events = []

        first = (not self.layer_time_epoch_avg_path.exists()) or (self.layer_time_epoch_avg_path.stat().st_size == 0)
        if first:
            with open(self.layer_time_epoch_avg_path, "a", newline="") as f:
                csv.writer(f).writerow(["epoch","x_label","phase","layer","avg_ms"])

    @torch.no_grad()
    def log_now(self, epoch: int, step: int, phase: str, layer_idx: int):
        mem = torch.cuda.memory_allocated(self.device)
        self.raw_writer.writerow([epoch, step, phase, layer_idx, f"{bytes_to_mib(mem):.3f}"])
        key = (phase, layer_idx)
        total, cnt = self.epoch_acc.get(key, (0.0, 0))
        self.epoch_acc[key] = (total + bytes_to_mib(mem), cnt + 1)

    def log_layer_time(self, epoch: int, step: int, phase: str, layer_idx: int, ms: float):
        self.time_writer.writerow([iso_now(), epoch, step, phase, layer_idx, f"{ms:.3f}"])
        key = (phase, layer_idx)
        total, cnt = self.epoch_time_acc.get(key, (0.0, 0))
        self.epoch_time_acc[key] = (total + ms, cnt + 1)

    def buffer_layer_time(self, epoch, step, phase, layer_idx, start_ev, end_ev):
        self.pending_events.append((epoch, step, phase, layer_idx, start_ev, end_ev))

    def process_buffered_events(self):
        if not self.pending_events:
            return
        torch.cuda.synchronize()
        for epoch, step, phase, layer_idx, start, end in self.pending_events:
            ms = start.elapsed_time(end)
            self.time_writer.writerow([iso_now(), epoch, step, phase, layer_idx, f"{ms:.3f}"])
            key = (phase, layer_idx)
            total, cnt = self.epoch_time_acc.get(key, (0.0, 0))
            self.epoch_time_acc[key] = (total + ms, cnt + 1)
        self.pending_events.clear()

    def reset_epoch_acc(self):
        self.epoch_acc.clear()
        self.epoch_time_acc.clear()
        self.pending_events.clear()

    def flush_epoch_avg(self, epoch: int):
        self.process_buffered_events()
        first_write = (not self.epoch_avg_path.exists()) or (self.epoch_avg_path.stat().st_size == 0)
        with open(self.epoch_avg_path, "a", newline="") as f:
            w = csv.writer(f)
            if first_write:
                w.writerow(["epoch","x_label","phase","layer","mem_mib"])
            for i in range(1, self.n_layers + 1):
                total, cnt = self.epoch_acc.get(("fwd", i), (0.0, 1))
                w.writerow([epoch, f"fwd-L{i}", "fwd", i, f"{(total/cnt):.3f}"])
            for i in range(self.n_layers, 0, -1):
                total, cnt = self.epoch_acc.get(("bwd", i), (0.0, 1))
                w.writerow([epoch, f"bwd-L{i}", "bwd", i, f"{(total/cnt):.3f}"])
        with open(self.layer_time_epoch_avg_path, "a", newline="") as f:
            w = csv.writer(f)
            for i in range(1, self.n_layers + 1):
                tot, cnt = self.epoch_time_acc.get(("fwd", i), (0.0, 0))
                if cnt > 0:
                    w.writerow([epoch, f"fwd-L{i}", "fwd", i, f"{(tot/cnt):.3f}"])
            for i in range(self.n_layers, 0, -1):
                tot, cnt = self.epoch_time_acc.get(("bwd", i), (0.0, 0))
                if cnt > 0:
                    w.writerow([epoch, f"bwd-L{i}", "bwd", i, f"{(tot/cnt):.3f}"])

    def close(self):
        try: self.raw_file.close()
        except Exception: pass
        try: self.time_file.close()
        except Exception: pass


class SmoothedValue:
    def __init__(self, momentum=0.98):
        self.m = None
        self.beta = momentum
    def update(self, x):
        self.m = x if self.m is None else self.beta * self.m + (1 - self.beta) * x
    @property
    def value(self):
        return float(self.m) if self.m is not None else float("nan")


# -------------------------------
# Eval
# -------------------------------
def run_eval(model, loader, device, pwr, epoch_idx=-1):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    total_loss = 0.0
    start_t = time.time()
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            torch.cuda.synchronize()
            p_start = pwr.sample_power_w()
            t0 = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * x.size(0)
            torch.cuda.synchronize()
            step_t = time.time() - t0
            p_end = pwr.sample_power_w()
            pwr.log_step("eval", epoch_idx, step, step_t, p_start, p_end)
    total_time = time.time() - start_t
    acc = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return acc, avg_loss, total_time


# -------------------------------
# Memory / Timing Hooks
# -------------------------------
def attach_mem_hooks(model, memlog, epoch_ref, step_ref):
    handles = []
    fwd_start_events = {}
    bwd_start_events = {}

    def make_fwd_pre(idx):
        def _hook(module, inp):
            memlog.log_now(epoch_ref(), step_ref(), "fwd", idx)
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(torch.cuda.current_stream())
            fwd_start_events[idx] = ev
        return _hook

    def make_fwd_end(idx):
        def _hook(module, inp, out):
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream())
            # Use deferred sync to avoid blocking GPU pipeline mid-forward
            start = fwd_start_events.pop(idx, None)
            if start is not None:
                memlog.buffer_layer_time(epoch_ref(), step_ref(), "fwd", idx, start, end)
        return _hook

    have_bwd_pre = hasattr(nn.Module, "register_full_backward_pre_hook")

    def make_bwd_pre(idx):
        def _hook(module, grad_input):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(torch.cuda.current_stream())
            bwd_start_events[idx] = ev
        return _hook

    def make_bwd_end(idx):
        def _hook(module, grad_input, grad_output):
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream())
            # Use deferred sync to avoid blocking GPU pipeline
            start = bwd_start_events.pop(idx, None)
            if start is not None:
                memlog.buffer_layer_time(epoch_ref(), step_ref(), "bwd", idx, start, end)
            memlog.log_now(epoch_ref(), step_ref(), "bwd", idx)
        return _hook

    blocks = list(model.layers)
    for i, block in enumerate(blocks, start=1):
        handles.append(block.register_forward_pre_hook(make_fwd_pre(i), with_kwargs=False))
        handles.append(block.register_forward_hook(make_fwd_end(i)))
        if have_bwd_pre:
            handles.append(block.register_full_backward_pre_hook(make_bwd_pre(i)))
        handles.append(block.register_full_backward_hook(make_bwd_end(i)))
    return handles


# -------------------------------
# Adaptive Dynamic Checkpointing
# -------------------------------
def inject_dynamic_checkpointing(model, device, mem_cap_bytes, step_ref, memlog, epoch_ref, pwr=None, threshold_ratio=0.8):
    blocks = list(model.layers)
    for layer_idx, block in enumerate(blocks, start=1):
        orig_forward = block.forward
        block._ckpt_last_step = -1
        block._use_ckpt_after = False
        block._in_recompute = False

        def make_forward(b, orig_fwd, li):
            def forward(x):
                if not torch.is_grad_enabled():
                    return orig_fwd(x)
                cur_step = step_ref()
                if getattr(b, "_ckpt_last_step", -1) != cur_step:
                    b._ckpt_last_step = cur_step
                    b._use_ckpt_after = False
                if getattr(b, "_in_recompute", False):
                    return orig_fwd(x)
                if not b._use_ckpt_after:
                    cur_bytes = torch.cuda.memory_allocated(device=device)
                    if cur_bytes >= threshold_ratio * mem_cap_bytes:
                        b._use_ckpt_after = True
                if not b._use_ckpt_after:
                    return orig_fwd(x)

                def run_block(inp):
                    memlog.log_now(epoch_ref(), step_ref(), "fwd_re", li)
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    start_ev.record(torch.cuda.current_stream())
                    t0 = time.time()
                    p_start = pwr.sample_power_w() if pwr else float("nan")
                    was = b._in_recompute
                    b._in_recompute = True
                    try:
                        out = orig_fwd(inp)
                    finally:
                        b._in_recompute = was
                    end_ev.record(torch.cuda.current_stream())
                    memlog.buffer_layer_time(epoch_ref(), step_ref(), "fwd_re", li, start_ev, end_ev)
                    if pwr:
                        pwr.log_step("train_fwd_re", epoch_ref(), step_ref(), time.time()-t0, p_start, pwr.sample_power_w())
                    return out
                return checkpoint(run_block, x, use_reentrant=False)
            return forward
        block.forward = make_forward(block, orig_forward, layer_idx)


# -------------------------------
# QLoRA quantization
# -------------------------------
def quantize_mamba_model_in_place(model):
    """Replace nn.Linear layers with bnb.nn.Linear4bit for 4-bit NF4 quantization.
    Skips: head (classification), dt_proj/dt_proj_b (accessed via .weight/.bias
    directly in ssm()), and Conv layers (conv1d, PatchEmbed)."""
    if not BNB_AVAILABLE:
        raise ImportError("bitsandbytes not installed. Needed for QLoRA.")
    # dt_proj layers are used via F.linear(x, dt_proj.weight) and dt_proj.bias
    # inside ssm(), so their weight must remain uncompressed.
    SKIP_NAMES = {"head", "dt_proj", "dt_proj_b"}
    print("[QLoRA] Quantizing Mamba model layers to 4-bit NF4...")

    def replace_linear(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            if any(s in full_name.split(".") for s in SKIP_NAMES):
                continue
            if isinstance(child, nn.Linear):
                new_layer = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.float16,
                    quant_type="nf4",
                )
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, new_layer)
            else:
                replace_linear(child, full_name)

    replace_linear(model)
    return model


# -------------------------------
# Training Logic
# -------------------------------
def train_single_run(args, dataset, peft_method, ckpt_mode):
    run_label = f"{dataset}/{peft_method}_{ckpt_mode}"
    print(f"\n{'='*80}\nSTARTING: {run_label}\n{'='*80}")
    set_seed(SEED)
    device = ensure_cuda()

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    cap_bytes = int(max(0.1, MEMORY_CAPACITY_GB) * (1024**3))
    try:
        if ckpt_mode == "adaptive" or ckpt_mode == "static":
            frac = min(0.99, cap_bytes / total_bytes)
            torch.cuda.set_per_process_memory_fraction(frac, device=0)
            print(f"[GPU MEM CAP] ~{MEMORY_CAPACITY_GB:.2f} GB ({frac*100:.1f}%)")
        else:
            torch.cuda.set_per_process_memory_fraction(0.99, device=0)
            print(f"[GPU MEM] Full memory (ckpt_mode={ckpt_mode})")
    except Exception:
        pass

    ab_vals = sorted(set(int(s) for s in args.sam_ab.split(",") if s.strip()))
    outdir = Path(args.outdir) / dataset / f"{peft_method}_{ckpt_mode}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN DIR] {outdir}")

    raw_log_gz = outdir / "memlog_raw.csv.gz"
    epoch_avg_csv = outdir / "memlog_epoch_avg.csv"
    layer_times_gz = outdir / "layer_times.csv.gz"
    layer_time_epoch_avg_csv = outdir / "layer_time_epoch_avg.csv"
    step_energy_gz = outdir / "step_energy.csv.gz"
    metrics_csv = outdir / "epoch_metrics.csv"
    ensure_metrics_csv_header(ab_vals, metrics_csv)

    # Data
    if dataset == "cifar100":
        train_loader, test_loader, num_classes = make_dataloaders_cifar100(
            args.steps_per_epoch, args.batch_size, NUM_WORKERS,
            args.data_root, True, args.eval_batch_size)
        print(f"[CIFAR-100] num_classes={num_classes}")
    else:
        train_loader, test_loader, num_classes = make_dataloaders_dtd(
            args.steps_per_epoch, args.batch_size, NUM_WORKERS,
            args.data_root, True, args.eval_batch_size)
        print(f"[DTD] num_classes={num_classes}")

    lr = args.lr_fullft if peft_method == "none" else args.lr
    print(f"[LR] {lr:.1e} ({'FullFT' if peft_method == 'none' else peft_method})")

    # Model
    mid_cls, bidir = False, False
    if args.pretrained:
        mid_cls, bidir = True, True
        print("Using Mid-Cls-Token & BiMamba for pretrained weights")

    model = VisionMamba(
        img_size=224, patch_size=16, depth=24, embed_dim=384,
        d_state=16, d_conv=4, expand=2, num_classes=num_classes,
        mid_cls_token=mid_cls, bidirectional=bidir)

    if args.pretrained:
        model = load_pretrained_weights(model, "vim_small_midclstok")

    # PEFT
    target_modules = ["in_proj", "x_proj", "dt_proj", "out_proj", "proj"]
    if peft_method == "qlora":
        model = quantize_mamba_model_in_place(model)
        model.to(device)
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r*2,
                            target_modules=target_modules, lora_dropout=0.1,
                            bias="none", modules_to_save=["head"])
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    elif peft_method == "lora":
        config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r*2,
                            target_modules=target_modules, lora_dropout=0.1,
                            bias="none", modules_to_save=["head"])
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    elif peft_method == "adalora":
        total_steps = args.epochs * args.steps_per_epoch
        config = AdaLoraConfig(r=args.lora_r, lora_alpha=args.lora_r*2,
                               target_modules=["in_proj","x_proj","dt_proj","out_proj"],
                               lora_dropout=0.1, bias="none", modules_to_save=["head"],
                               init_r=12, tinit=200, tfinal=1000, deltaT=10,
                               total_step=total_steps)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    elif peft_method == "bitfit":
        for p in model.parameters(): p.requires_grad = False
        cnt_t, cnt_all = 0, 0
        for n, p in model.named_parameters():
            if "bias" in n or "head" in n:
                p.requires_grad = True
                cnt_t += p.numel()
            cnt_all += p.numel()
        print(f"BitFit: {cnt_t/1e6:.2f}M / {cnt_all/1e6:.2f}M ({cnt_t/cnt_all*100:.2f}%)")
    else:
        print(f"Full Fine-Tuning: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    use_static_ckpt = (ckpt_mode == "static")
    if use_static_ckpt:
        print("[Checkpoint] Static gradient checkpointing enabled")

    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    cur_epoch = {"v": 0}
    cur_step = {"v": 0}
    global_step = {"v": 0}
    epoch_ref = lambda: cur_epoch["v"]
    step_ref = lambda: cur_step["v"]

    base_model = model.base_model.model if hasattr(model, "base_model") and hasattr(model.base_model, "model") else model
    n_blocks = len(list(base_model.layers))
    print(f"Found {n_blocks} Vim blocks")

    memlog = MemLogger(device, n_blocks, raw_log_gz, epoch_avg_csv, layer_times_gz, layer_time_epoch_avg_csv)
    pwr = GpuPowerMeter(device_index=0, step_energy_path=step_energy_gz)

    if ckpt_mode == "adaptive":
        inject_dynamic_checkpointing(base_model, device=device, mem_cap_bytes=cap_bytes,
                                     step_ref=step_ref, memlog=memlog, epoch_ref=epoch_ref,
                                     pwr=pwr, threshold_ratio=0.50)

    handles = attach_mem_hooks(base_model, memlog, epoch_ref, step_ref)

    try:
        # Baseline
        pwr.reset_epoch()
        base_acc, base_loss, base_time = run_eval(model, test_loader, device, pwr, epoch_idx=0)
        print(f"[BASELINE] Acc={base_acc:.2f}% Loss={base_loss:.4f} Time={base_time:.2f}s")
        with open(metrics_csv, "a", newline="") as f:
            row = [0, 0.0, f"{base_time:.3f}", f"{base_time:.3f}", 0.0, 0.0, 0.0, 0.0, f"{base_acc:.2f}"]
            for _ in ab_vals: row.append("nan")
            csv.writer(f).writerow(row)

        # Free Triton/eval workspace before training allocates activation graph
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        for epoch in range(1, args.epochs + 1):
            cur_epoch["v"] = epoch
            memlog.reset_epoch_acc()
            pwr.reset_epoch()
            model.train()
            torch.cuda.reset_peak_memory_stats(device)
            loss_smooth = SmoothedValue(0.98)
            t_epoch = time.time()

            iterator = enumerate(train_loader, start=1)
            if not args.no_progress:
                iterator = tqdm(iterator, total=args.steps_per_epoch, ncols=120, leave=False, desc=f"Epoch {epoch}")

            for step, (x, y) in iterator:
                cur_step["v"] = step
                torch.cuda.synchronize()
                p_start = pwr.sample_power_w()
                t0 = time.time()
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                # AdaLoRA's update_ipt does `p * p.grad` and crashes if grad is None.
                # set_to_none=False keeps zero tensors for params not in the compute graph.
                optimizer.zero_grad(set_to_none=(peft_method != "adalora"))

                if use_static_ckpt:
                    x_feat = base_model.patch_embed(x)
                    B, N, C = x_feat.shape
                    if base_model.mid_cls_token:
                        cls_tok = base_model.cls_token.expand(B, -1, -1)
                        mid = N // 2
                        x_feat = torch.cat((x_feat[:,:mid,:], cls_tok, x_feat[:,mid:,:]), dim=1)
                    else:
                        cls_tok = base_model.cls_token.expand(B, -1, -1)
                        x_feat = torch.cat((cls_tok, x_feat), dim=1)
                    x_feat = x_feat + base_model.pos_embed
                    x_feat = base_model.pos_drop(x_feat)
                    for layer in base_model.layers:
                        x_feat = checkpoint(layer, x_feat, use_reentrant=False)
                    x_feat = base_model.norm_f(x_feat)
                    if base_model.mid_cls_token:
                        x_feat = x_feat[:, base_model.patch_embed.num_patches // 2]
                    else:
                        x_feat = x_feat[:, 0]
                    out = base_model.head(x_feat) if not hasattr(model, "base_model") else model.head(x_feat)
                    loss = criterion(out, y)
                else:
                    if args.amp:
                        with torch.cuda.amp.autocast():
                            out = model(x)
                            loss = criterion(out, y)
                    else:
                        out = model(x)
                        loss = criterion(out, y)

                if args.amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                global_step["v"] += 1

                if peft_method == "adalora" and hasattr(model, "base_model") and hasattr(model.base_model, "update_and_allocate"):
                    model.base_model.update_and_allocate(global_step["v"])

                torch.cuda.synchronize()
                step_t = time.time() - t0
                p_end = pwr.sample_power_w()
                pwr.log_step("train", epoch, step, step_t, p_start, p_end)
                loss_smooth.update(loss.item())
                memlog.process_buffered_events()

                if step % args.log_interval == 0:
                    alloc = bytes_to_mib(torch.cuda.memory_allocated(device))
                    peak = bytes_to_mib(torch.cuda.max_memory_allocated(device))
                    print(f"[E{epoch:02d} S{step:05d}] loss={loss.item():.4f} sm={loss_smooth.value:.4f} alloc={alloc:.0f}MiB peak={peak:.0f}MiB")

                if step >= args.steps_per_epoch:
                    break

            dt_train = time.time() - t_epoch
            acc, val_loss, dt_eval = run_eval(model, test_loader, device, pwr, epoch_idx=epoch)
            memlog.flush_epoch_avg(epoch)
            totals = pwr.epoch_totals()
            sam_vals = compute_sam(acc, totals["total_energy_j"], ab_vals)
            peak_epoch = bytes_to_mib(torch.cuda.max_memory_allocated(device))

            print(f"[Epoch {epoch}/{args.epochs}] Acc={acc:.2f}% TrainT={dt_train/60:.1f}m EvalT={dt_eval:.1f}s Energy={totals['total_energy_j']:.0f}J Peak={peak_epoch:.0f}MiB")

            row = [epoch, f"{totals['train_time_s']:.3f}", f"{totals['eval_time_s']:.3f}",
                   f"{totals['total_time_s']:.3f}", f"{totals['train_energy_j']:.3f}",
                   f"{totals['eval_energy_j']:.3f}", f"{totals['total_energy_j']:.3f}",
                   f"{totals['avg_power_w']:.3f}", f"{acc:.2f}"]
            for a in ab_vals:
                v = sam_vals[f"SAM_a{a}_b{a}"]
                row.append(f"{v:.6f}" if not math.isnan(v) else "nan")
            with open(metrics_csv, "a", newline="") as f:
                csv.writer(f).writerow(row)

    finally:
        for h in handles: h.remove()
        memlog.close()
        pwr.close()
        torch.cuda.empty_cache()


# -------------------------------
# Args & Multi-Experiment Runner
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Vim-Small: DTD/CIFAR-100 with PEFT + adaptive checkpointing")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4, help="PEFT learning rate")
    ap.add_argument("--lr-fullft", type=float, default=1e-5, help="FullFT learning rate")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--sam-ab", type=str, default="1,2,3,4,5")
    ap.add_argument("--datasets", nargs="+", default=["cifar100","dtd"], choices=["cifar100","dtd"])
    ap.add_argument("--peft-methods", nargs="+", default=["none","bitfit"],
                    choices=["none","bitfit","lora","qlora","adalora"])
    ap.add_argument("--ckpt-modes", nargs="+", default=["static"],
                    choices=["none","static","adaptive"])
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--pretrained", action="store_true", help="Load official Vim-Small-MidCls weights")
    return ap.parse_args()


def main():
    args = parse_args()
    ds_list = args.datasets
    peft_list = args.peft_methods
    ckpt_list = args.ckpt_modes
    total = len(ds_list) * len(peft_list) * len(ckpt_list)

    print(f"\n{'#'*80}")
    print(f"EXPERIMENT MATRIX: {total} runs")
    print(f"  Datasets:    {ds_list}")
    print(f"  PEFT:        {peft_list}")
    print(f"  Checkpoints: {ckpt_list}")
    print(f"  Pretrained:  {args.pretrained}")
    print(f"{'#'*80}\n")

    completed, failed = [], []
    idx = 0
    for ds in ds_list:
        for peft in peft_list:
            for ckpt in ckpt_list:
                idx += 1
                label = f"{ds}/{peft}_{ckpt}"
                print(f"\n>>> [{idx}/{total}] {label}")
                try:
                    train_single_run(args, dataset=ds, peft_method=peft, ckpt_mode=ckpt)
                    completed.append(label)
                except Exception as e:
                    print(f"\n[ERROR] '{label}' failed: {e}")
                    traceback.print_exc()
                    failed.append(label)
                torch.cuda.empty_cache()

    print(f"\n{'='*80}\nEXPERIMENT SUMMARY\n{'='*80}")
    print(f"Completed: {len(completed)}/{total}")
    for c in completed: print(f"  + {c}")
    if failed:
        print(f"Failed: {len(failed)}/{total}")
        for f in failed: print(f"  x {f}")



if __name__ == "__main__":
    main()

