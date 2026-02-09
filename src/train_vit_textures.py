import os

# Kaggle-friendly dirs
KAGGLE_WORKING = "/kaggle/working"
KAGGLE_INPUT = "/kaggle/input"

# Оптимизация аллокатора памяти
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

import math
import time
import gzip
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import copy

from torch.utils.checkpoint import checkpoint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torchvision import datasets, transforms

import timm
from tqdm import tqdm

# --- PEFT / bitsandbytes ---
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
# Defaults
# -------------------------------
EPOCHS = 10
STEPS_PER_EPOCH = 782
BATCH_SIZE = 64
NUM_WORKERS = 2  # Kaggle обычно стабильно на 2-4
MODEL_NAME = "vit_small_patch16_224"
SEED = 42
MEMORY_CAPACITY_GB = 2.0
DEFAULT_OUTDIR = Path('./texture_results')
DEFAULT_DATA_ROOT = Path('./data')
GPU_INDEX = 0


def iso_now():
    return datetime.now().isoformat(timespec="seconds")


def set_seed(seed=SEED):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def bytes_to_mib(x: int) -> float:
    return x / (1024.0 * 1024.0)


def ensure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU не обнаружена.")
    print("Тренировка с GPU")
    return torch.device("cuda")


# -------------------------------
# DTD Dataloaders
# -------------------------------
def make_dataloaders_dtd(steps_per_epoch: int, batch_size: int, num_workers: int, data_root: str, download: bool):
    """
    DTD (Describable Textures Dataset)
    split: train/val/test
    обычно train+val -> train, test -> eval
    """
    img_size = 224
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

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

    # DTD: 47 классов
    dtd_train = datasets.DTD(root=str(root), split="train", transform=train_tf, download=download)
    dtd_val   = datasets.DTD(root=str(root), split="val",   transform=train_tf, download=download)
    dtd_test  = datasets.DTD(root=str(root), split="test",  transform=test_tf,  download=download)

    num_classes = len(dtd_train.classes)

    train_ds = ConcatDataset([dtd_train, dtd_val])
    test_ds = dtd_test

    # фиксируем steps_per_epoch через RandomSampler
    num_samples = steps_per_epoch * batch_size
    train_sampler = RandomSampler(train_ds, replacement=True, num_samples=num_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, num_classes


# -------------------------------
# QLoRA quantize timm (Linear -> Linear4bit), skip head
# -------------------------------
def quantize_timm_model_in_place(model):
    if not BNB_AVAILABLE:
        raise ImportError("bitsandbytes not installed. Needed for QLoRA.")
    print("[QLoRA] Quantizing model layers to 4-bit...")

    def replace_linear(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            if "head" in full_name:
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


def get_base_model_from_peft(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


# -------------------------------
# Build model + PEFT
# -------------------------------
def build_model(num_classes: int, ckpt: bool, peft_method: str, args):
    print(f"Building model: {MODEL_NAME} | Mode: {peft_method}")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)

    if peft_method == "bitfit":
        print("[BitFit] Freezing weights, enabling biases...")
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "bias" in name or "head" in name:
                param.requires_grad = True

    elif peft_method == "qlora":
        if not PEFT_AVAILABLE:
            raise ImportError("peft required for QLoRA")
        model = quantize_timm_model_in_place(model)
        model = prepare_model_for_kbit_training(model)

    if peft_method in ["lora", "qlora", "adalora"]:
        if not PEFT_AVAILABLE:
            raise ImportError("peft required")

        target_modules = [t.strip() for t in args.lora_targets.split(",") if t.strip()]

        # Важно: AdaLora в peft не поддерживает Conv2d (а в ViT есть patch_embed.proj = Conv2d).
        # Поэтому для adalora убираем 'proj', чтобы не матчиться на patch_embed.proj.
        if peft_method == "adalora" and "proj" in target_modules:
            target_modules = [t for t in target_modules if t != "proj"]
            print(f"[AdaLora] target_modules adjusted: {target_modules}")

        common_args = dict(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            modules_to_save=["head"],
        )

        if peft_method == "adalora":
            total_steps = args.epochs * args.steps_per_epoch
            config = AdaLoraConfig(
                **common_args,
                total_step=total_steps,
                target_r=args.adalora_target_r,
                init_r=args.adalora_init_r,
                tinit=args.adalora_tinit,
                tfinal=args.adalora_tfinal,
                deltaT=args.adalora_deltaT,
            )
        else:
            config = LoraConfig(**common_args)

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    base = get_base_model_from_peft(model)
    assert hasattr(base, "blocks"), "Could not find .blocks in model"

    if ckpt and peft_method != "qlora" and hasattr(base, "set_grad_checkpointing"):
        base.set_grad_checkpointing(enable=True)
        print("Static checkpoint enabled (native)")

    return model


# -------------------------------
# Power/Energy logger
# -------------------------------
class GpuPowerMeter:
    def __init__(self, device_index: int, step_energy_path: Path):
        self.available = False
        self.handle = None
        self.device_index = device_index
        self._init_nvml()

        self.reset_epoch()
        self._step_file = gzip.open(step_energy_path, "at", newline="")
        self._step_writer = csv.writer(self._step_file)
        if step_energy_path.stat().st_size == 0:
            self._step_writer.writerow(["ts","epoch","step","phase","step_ms","p_start_w","p_end_w","p_avg_w","energy_j"])

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
        if not self.available:
            return float("nan")
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
        total_e = total_e if (not math.isnan(self.train_energy_j) or not math.isnan(self.eval_energy_j)) else float("nan")
        total_t = self.train_time_s + self.eval_time_s
        avg_power = (total_e / total_t) if (not math.isnan(total_e) and total_t > 0) else float("nan")
        return dict(
            train_energy_j=self.train_energy_j, eval_energy_j=self.eval_energy_j,
            total_energy_j=total_e, train_time_s=self.train_time_s, eval_time_s=self.eval_time_s,
            total_time_s=total_t, avg_power_w=avg_power
        )


# -------------------------------
# Memory Logger (per-run files)
# -------------------------------
class MemLogger:
    def __init__(self, device, n_layers: int, raw_log_gz: Path, epoch_avg_csv: Path, layer_times_gz: Path, layer_time_epoch_avg_csv: Path):
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

    def reset_epoch_acc(self):
        self.epoch_acc.clear()
        self.epoch_time_acc.clear()

    def flush_epoch_avg(self, epoch: int):
        first_write = (not self.epoch_avg_path.exists()) or (self.epoch_avg_path.stat().st_size == 0)
        with open(self.epoch_avg_path, "a", newline="") as f:
            w = csv.writer(f)
            if first_write:
                w.writerow(["epoch","x_label","phase","layer","mem_mib"])
            for i in range(1, self.n_layers + 1):
                total, cnt = self.epoch_acc.get(("fwd", i), (0.0, 1))
                w.writerow([epoch, f"fwd-L{i}", "fwd", i, f"{(total / cnt):.3f}"])
            for i in range(self.n_layers, 0, -1):
                total, cnt = self.epoch_acc.get(("bwd", i), (0.0, 1))
                w.writerow([epoch, f"bwd-L{i}", "bwd", i, f"{(total / cnt):.3f}"])

        with open(self.layer_time_epoch_avg_path, "a", newline="") as f:
            w = csv.writer(f)
            for i in range(1, self.n_layers + 1):
                tot, cnt = self.epoch_time_acc.get(("fwd", i), (0.0, 0))
                if cnt > 0:
                    w.writerow([epoch, f"fwd-L{i}", "fwd", i, f"{(tot / cnt):.3f}"])
            for i in range(self.n_layers, 0, -1):
                tot, cnt = self.epoch_time_acc.get(("bwd", i), (0.0, 0))
                if cnt > 0:
                    w.writerow([epoch, f"bwd-L{i}", "bwd", i, f"{(tot / cnt):.3f}"])

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
            header = ["epoch","train_time_s","eval_time_s","total_time_s","train_energy_j","eval_energy_j","total_energy_j","avg_power_w","test_acc_pct"]
            for a in ab_values:
                header.append(f"SAM_a{a}_b{a}")
            w.writerow(header)


def run_eval(model, loader, device, pwr: GpuPowerMeter, epoch_idx=-1):
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


def attach_mem_hooks(model: nn.Module, memlog: MemLogger, epoch_ref, step_ref):
    handles = []
    fwd_start_events: Dict[int, torch.cuda.Event] = {}
    bwd_start_events: Dict[int, torch.cuda.Event] = {}

    def make_fwd_pre(layer_idx):
        def _hook(module, inp):
            memlog.log_now(epoch_ref(), step_ref(), "fwd", layer_idx)
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(torch.cuda.current_stream())
            fwd_start_events[layer_idx] = ev
        return _hook

    def make_fwd_end(layer_idx):
        def _hook(module, inp, out):
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream())
            end.synchronize()
            start = fwd_start_events.pop(layer_idx, None)
            if start is not None:
                ms = start.elapsed_time(end)
                memlog.log_layer_time(epoch_ref(), step_ref(), "fwd", layer_idx, ms)
        return _hook

    have_bwd_pre = hasattr(nn.Module, "register_full_backward_pre_hook")

    def make_bwd_pre(layer_idx):
        def _hook(module, grad_input):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(torch.cuda.current_stream())
            bwd_start_events[layer_idx] = ev
        return _hook

    def make_bwd_end(layer_idx):
        def _hook(module, grad_input, grad_output):
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream())
            end.synchronize()
            start = bwd_start_events.pop(layer_idx, None)
            ms = start.elapsed_time(end) if start is not None else 0.0
            memlog.log_now(epoch_ref(), step_ref(), "bwd", layer_idx)
            memlog.log_layer_time(epoch_ref(), step_ref(), "bwd", layer_idx, ms)
        return _hook

    for i, block in enumerate(model.blocks, start=1):
        handles.append(block.register_forward_pre_hook(make_fwd_pre(i), with_kwargs=False))
        handles.append(block.register_forward_hook(make_fwd_end(i)))
        if have_bwd_pre:
            handles.append(block.register_full_backward_pre_hook(make_bwd_pre(i)))
            handles.append(block.register_full_backward_hook(make_bwd_end(i)))
        else:
            handles.append(block.register_full_backward_hook(make_bwd_end(i)))
    return handles


def inject_dynamic_checkpointing(model: nn.Module, device: torch.device, mem_cap_bytes: int, step_ref, memlog: MemLogger, epoch_ref, pwr: GpuPowerMeter = None, threshold_ratio: float = 0.8):
    for layer_idx, block in enumerate(model.blocks, start=1):
        orig_forward = block.forward
        block._ckpt_last_step = -1
        block._use_ckpt_after = False
        block._in_recompute = False

        def make_forward(b, orig_fwd, layer_idx):
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
                    memlog.log_now(epoch_ref(), step_ref(), "fwd_re", layer_idx)
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)
                    start_ev.record(torch.cuda.current_stream())

                    t0 = time.time()
                    p_start = pwr.sample_power_w() if pwr is not None else float("nan")

                    was_flag = b._in_recompute
                    b._in_recompute = True
                    try:
                        out = orig_fwd(inp)
                    finally:
                        b._in_recompute = was_flag

                    end_ev.record(torch.cuda.current_stream())
                    end_ev.synchronize()
                    ms = start_ev.elapsed_time(end_ev)
                    memlog.log_layer_time(epoch_ref(), step_ref(), "fwd_re", layer_idx, ms)

                    if pwr is not None:
                        step_t = time.time() - t0
                        p_end = pwr.sample_power_w()
                        pwr.log_step("train_fwd_re", epoch_ref(), step_ref(), step_t, p_start, p_end)

                    return out

                return checkpoint(run_block, x, use_reentrant=False)
            return forward

        block.forward = make_forward(block, orig_forward, layer_idx)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--gpu-index", type=int, default=GPU_INDEX)

    ap.add_argument("--sam-ab", type=str, default="1,2,3,4,5")
    ap.add_argument("--ckpt-mode", type=str, default="adaptive", choices=["none","static","adaptive"])

    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--download", action="store_true", help="Download DTD dataset (needs Kaggle Internet=On)")

    ap.add_argument("--peft-method", type=str, default="none",
                    choices=["none","lora","qlora","adalora","bitfit","all"])

    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    # По умолчанию без 'proj' (чтобы точно не упасть на Conv2d patch_embed.proj)
    ap.add_argument("--lora-targets", type=str, default="qkv,fc1,fc2")

    ap.add_argument("--adalora-init-r", type=int, default=12)
    ap.add_argument("--adalora-target-r", type=int, default=8)
    ap.add_argument("--adalora-tinit", type=int, default=200)
    ap.add_argument("--adalora-tfinal", type=int, default=1000)
    ap.add_argument("--adalora-deltaT", type=int, default=10)

    return ap.parse_args()


def train(args):
    set_seed(SEED)
    device = ensure_cuda()

    # ---- GPU memory cap ----
    total_bytes = torch.cuda.get_device_properties(args.gpu_index).total_memory
    cap_bytes = int(max(0.1, MEMORY_CAPACITY_GB) * (1024**3))
    try:
        frac = min(0.99, cap_bytes / total_bytes)
        torch.cuda.set_per_process_memory_fraction(frac, device=args.gpu_index)
        print(f"[GPU MEM CAP] Limiting allocator to ~{MEMORY_CAPACITY_GB:.2f} GB ({frac*100:.1f}%).")
    except Exception:
        pass

    ab_vals = sorted(set(int(s) for s in args.sam_ab.split(",") if s.strip()))

    outdir_base = Path(args.outdir)
    outdir_base.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.peft_method}_{args.ckpt_mode}"
    run_dir = outdir_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN DIR] {run_dir}")

    raw_log_gz = run_dir / "memlog_raw.csv.gz"
    epoch_avg_csv = run_dir / "memlog_epoch_avg.csv"
    layer_times_gz = run_dir / "layer_times.csv.gz"
    layer_time_epoch_avg_csv = run_dir / "layer_time_epoch_avg.csv"
    step_energy_gz = run_dir / "step_energy.csv.gz"
    metrics_csv_path = run_dir / "epoch_metrics.csv"

    ensure_metrics_csv_header(ab_vals, metrics_csv_path)

    train_loader, test_loader, num_classes = make_dataloaders_dtd(
        args.steps_per_epoch, args.batch_size, args.num_workers,
        data_root=args.data_root,
        download=args.download,
    )
    print(f"[DTD] num_classes={num_classes}")

    use_static_ckpt = (args.ckpt_mode == "static")
    model = build_model(num_classes, ckpt=use_static_ckpt, peft_method=args.peft_method, args=args).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    cur_epoch = {"v": 0}
    cur_step = {"v": 0}
    epoch_ref = lambda: cur_epoch["v"]
    step_ref = lambda: cur_step["v"]

    base_model_for_hooks = get_base_model_from_peft(model)
    memlog = MemLogger(device, len(base_model_for_hooks.blocks), raw_log_gz, epoch_avg_csv, layer_times_gz, layer_time_epoch_avg_csv)
    pwr = GpuPowerMeter(device_index=args.gpu_index, step_energy_path=step_energy_gz)

    if args.ckpt_mode == "adaptive":
        inject_dynamic_checkpointing(
            base_model_for_hooks, device=device, mem_cap_bytes=cap_bytes,
            step_ref=step_ref, memlog=memlog, epoch_ref=epoch_ref,
            pwr=pwr, threshold_ratio=0.50
        )

    handles = attach_mem_hooks(base_model_for_hooks, memlog, epoch_ref, step_ref)

    try:
        print(">>> Baseline eval...")
        pwr.reset_epoch()
        base_acc, base_loss, base_time = run_eval(model, test_loader, device, pwr, epoch_idx=0)
        print(f"[BASELINE] Acc={base_acc:.2f}% Loss={base_loss:.4f} Time={base_time:.2f}s")

        with open(metrics_csv_path, "a", newline="") as f:
            row = [0, 0.0, f"{base_time:.3f}", f"{base_time:.3f}", 0.0, 0.0, 0.0, 0.0, f"{base_acc:.2f}"]
            for _ in ab_vals: row.append("nan")
            csv.writer(f).writerow(row)

        for epoch in range(1, args.epochs + 1):
            cur_epoch["v"] = epoch
            memlog.reset_epoch_acc()
            pwr.reset_epoch()

            model.train()
            torch.cuda.reset_peak_memory_stats(device)
            loss_smooth = SmoothedValue(0.98)
            start_epoch_t = time.time()

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

                optimizer.zero_grad(set_to_none=True)
                if args.amp:
                    with torch.cuda.amp.autocast():
                        out = model(x)
                        loss = criterion(out, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(x)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()

                torch.cuda.synchronize()
                step_t = time.time() - t0
                p_end = pwr.sample_power_w()
                pwr.log_step("train", epoch, step, step_t, p_start, p_end)
                loss_smooth.update(loss.item())

                if step % args.log_interval == 0:
                    alloc = bytes_to_mib(torch.cuda.memory_allocated(device))
                    peak = bytes_to_mib(torch.cuda.max_memory_allocated(device))
                    lr_curr = optimizer.param_groups[0]["lr"]
                    print(f"[E{epoch:02d} S{step:05d}] loss={loss.item():.4f} sm={loss_smooth.value:.4f} lr={lr_curr:.2e} alloc={alloc:.0f}MiB peak={peak:.0f}MiB")

                if step >= args.steps_per_epoch:
                    break

            dt_train = time.time() - start_epoch_t
            acc, val_loss, dt_eval = run_eval(model, test_loader, device, pwr, epoch_idx=epoch)

            memlog.flush_epoch_avg(epoch)
            totals = pwr.epoch_totals()
            sam_vals = compute_sam(acc, totals["total_energy_j"], ab_vals)
            peak_epoch = bytes_to_mib(torch.cuda.max_memory_allocated(device))

            print(f"[Epoch {epoch}/{args.epochs}] Acc={acc:.2f}% TrainT={dt_train/60:.1f}m EvalT={dt_eval:.1f}s AvgW={totals['avg_power_w']:.1f} Energy={totals['total_energy_j']:.0f}J PeakMem={peak_epoch:.0f}MiB")

            row = [
                epoch,
                f"{totals['train_time_s']:.3f}",
                f"{totals['eval_time_s']:.3f}",
                f"{totals['total_time_s']:.3f}",
                f"{totals['train_energy_j']:.3f}",
                f"{totals['eval_energy_j']:.3f}",
                f"{totals['total_energy_j']:.3f}",
                f"{totals['avg_power_w']:.3f}",
                f"{acc:.2f}",
            ]
            for a in ab_vals:
                key = f"SAM_a{a}_b{a}"
                v = sam_vals[key]
                row.append(f"{v:.6f}" if not math.isnan(v) else "nan")

            with open(metrics_csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

    finally:
        for h in handles:
            h.remove()
        memlog.close()
        pwr.close()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    if args.peft_method == "all":
        methods = ["none", "bitfit", "lora", "adalora", "qlora"]
        for m in methods:
            print("=" * 80)
            print(f"Training PEFT method: {m}")
            print("=" * 80)
            args_run = copy.deepcopy(args)
            args_run.peft_method = m
            train(args_run)
    else:
        train(args)