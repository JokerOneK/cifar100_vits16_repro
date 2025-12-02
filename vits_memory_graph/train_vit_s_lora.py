import os

os.environ[
    "PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"

import math
import time
import gzip
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from torch.utils.checkpoint import checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms

# pip install timm tqdm peft
import timm
from tqdm import tqdm

# [NEW - LORA] Импорт PEFT
try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: 'peft' library not found. LoRA will not work unless installed.")

# -------------------------------
# Константы по умолчанию
# -------------------------------
EPOCHS = 10
STEPS_PER_EPOCH = 782
BATCH_SIZE = 64
NUM_WORKERS = 4
MODEL_NAME = "vit_small_patch16_224"
NUM_CLASSES = 100
SEED = 42
OUTDIR = Path("./mem_logs")
OUTDIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = True
MEMORY_CAPACITY_GB = 2.0
GPU_INDEX = 0

RAW_LOG_GZ = OUTDIR / "memlog_raw.csv.gz"
EPOCH_AVG_CSV = OUTDIR / "memlog_epoch_avg.csv"
PLOT_PNG = OUTDIR / "memplot_epochs.png"

LAYER_TIMES_GZ = OUTDIR / "layer_times.csv.gz"
LAYER_TIME_EPOCH_AVG_CSV = OUTDIR / "layer_time_epoch_avg.csv"
STEP_ENERGY_GZ = OUTDIR / "step_energy.csv.gz"
METRICS_CSV = OUTDIR / "epoch_metrics.csv"


def iso_now():
    return datetime.now().isoformat(timespec="seconds")


def set_seed(seed=SEED):
    import random, numpy as np
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
        print('Тренировка без GPU')
        raise RuntimeError("CUDA GPU не обнаружена.")
    print('Тренировка с GPU')
    return torch.device("cuda")


def make_dataloaders(steps_per_epoch: int, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ])

    train_ds = datasets.CIFAR100(root="./data", train=True, transform=train_tf, download=True)
    test_ds = datasets.CIFAR100(root="./data", train=False, transform=test_tf, download=True)

    num_samples = steps_per_epoch * batch_size
    train_sampler = RandomSampler(train_ds, replacement=True, num_samples=num_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


# [MODIFIED] Добавлена логика LoRA
def build_model(num_classes: int, ckpt: bool = False, use_lora: bool = False, lora_args=None):
    # 1. Создаем базовую модель timm (с новой головой под num_classes)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)

    # 2. Если включен LoRA
    if use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("Please install peft: `pip install peft`")

        print(f"[LoRA] Applying LoRA (rank={lora_args.lora_r}, alpha={lora_args.lora_alpha})...")

        # Разбираем целевые модули (обычно qkv, proj, fc1, fc2 для ViT)
        target_modules = [t.strip() for t in lora_args.lora_targets.split(',')]

        config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias="none",
            # ВАЖНО: 'head' обучаем полностью, т.к. классы ImageNet != CIFAR100
            modules_to_save=["head"]
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    # 3. Проверка блоков (для ViT)
    # Если модель обернута в PeftModel, блоки лежат глубже.
    # Эта проверка чисто информативная, доступ к блокам получим через get_base_model_from_peft
    base = get_base_model_from_peft(model)
    assert hasattr(base, "blocks") and len(base.blocks) == 12, "Ожидались 12 блоков в ViT-S/16"

    # Static Checkpoint (native timm implementation)
    # Если включен LoRA, native checkpointing может конфликтовать или требовать use_reentrant=False
    if ckpt and hasattr(base, "set_grad_checkpointing"):
        base.set_grad_checkpointing(enable=True)
        print(f'Статический чекпоинт включен (native)')

    return model


# [NEW] Вспомогательная функция для извлечения timm-модели из PeftModel
def get_base_model_from_peft(model):
    """
    Если model это PeftModel, возвращает model.base_model.model.
    Иначе возвращает model.
    """
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


# -------------------------------
# Power/Energy logger via NVML
# -------------------------------
class GpuPowerMeter:
    def __init__(self, device_index: int = 0):
        self.available = False
        self.handle = None
        self.device_index = device_index
        self._init_nvml()

        self.reset_epoch()
        self._step_file = gzip.open(STEP_ENERGY_GZ, "at", newline="")
        self._step_writer = csv.writer(self._step_file)
        if STEP_ENERGY_GZ.stat().st_size == 0:
            self._step_writer.writerow(["ts", "epoch", "step", "phase", "step_ms",
                                        "p_start_w", "p_end_w", "p_avg_w", "energy_j"])

    def _init_nvml(self):
        try:
            import pynvml
            self.nvml = pynvml
            self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(self.device_index)
            _ = self.nvml.nvmlDeviceGetPowerUsage(self.handle)
            self.available = True
        except Exception as e:
            self.available = False
            self.nvml = None
            self.handle = None
            print(f"[PowerMeter] NVML недоступен ({e}). Метрики мощности/энергии = NaN.")

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

        if phase == "train":
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
            train_energy_j=self.train_energy_j,
            eval_energy_j=self.eval_energy_j,
            total_energy_j=total_e,
            train_time_s=self.train_time_s,
            eval_time_s=self.eval_time_s,
            total_time_s=total_t,
            avg_power_w=avg_power
        )


class MemLogger:
    def __init__(self, device, n_layers: int):
        self.device = device
        self.n_layers = n_layers

        self.raw_file = gzip.open(RAW_LOG_GZ, "at", newline="")
        self.raw_writer = csv.writer(self.raw_file)
        if RAW_LOG_GZ.stat().st_size == 0:
            self.raw_writer.writerow(["epoch", "step", "phase", "layer", "mem_mib"])
        self.epoch_acc: Dict[Tuple[str, int], Tuple[float, int]] = {}

        self.time_file = gzip.open(LAYER_TIMES_GZ, "at", newline="")
        self.time_writer = csv.writer(self.time_file)
        if LAYER_TIMES_GZ.stat().st_size == 0:
            self.time_writer.writerow(["ts", "epoch", "step", "phase", "layer", "ms"])

        self.epoch_time_acc: Dict[Tuple[str, int], Tuple[float, int]] = {}

        first = not LAYER_TIME_EPOCH_AVG_CSV.exists() or LAYER_TIME_EPOCH_AVG_CSV.stat().st_size == 0
        if first:
            with open(LAYER_TIME_EPOCH_AVG_CSV, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "x_label", "phase", "layer", "avg_ms"])

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
        first_write = not EPOCH_AVG_CSV.exists() or EPOCH_AVG_CSV.stat().st_size == 0
        with open(EPOCH_AVG_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if first_write:
                w.writerow(["epoch", "x_label", "phase", "layer", "mem_mib"])
            for i in range(1, self.n_layers + 1):
                total, cnt = self.epoch_acc.get(("fwd", i), (0.0, 1))
                w.writerow([epoch, f"fwd-L{i}", "fwd", i, f"{(total / cnt):.3f}"])
            for i in range(self.n_layers, 0, -1):
                total, cnt = self.epoch_acc.get(("bwd", i), (0.0, 1))
                w.writerow([epoch, f"bwd-L{i}", "bwd", i, f"{(total / cnt):.3f}"])

        with open(LAYER_TIME_EPOCH_AVG_CSV, "a", newline="") as f:
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
        try:
            self.raw_file.close()
        except Exception:
            pass
        try:
            self.time_file.close()
        except Exception:
            pass


class SmoothedValue:
    def __init__(self, momentum=0.98):
        self.m = None
        self.beta = momentum

    def update(self, x):
        self.m = x if self.m is None else self.beta * self.m + (1 - self.beta) * x

    @property
    def value(self):
        return float(self.m) if self.m is not None else float("nan")


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
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--sam-ab", type=str, default="1,2,3,4,5")
    ap.add_argument(
        "--ckpt-mode",
        type=str,
        default=("adaptive" if CHECKPOINT else "none"),
        choices=["none", "static", "adaptive"]
    )
    # [NEW] Параметры LoRA
    ap.add_argument("--use-lora", action="store_true", help="Включить LoRA")
    ap.add_argument("--lora-r", type=int, default=8, help="Ранг LoRA (r)")
    ap.add_argument("--lora-alpha", type=int, default=16, help="Scaling factor (alpha)")
    ap.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    ap.add_argument("--lora-targets", type=str, default="qkv,proj,fc1,fc2",
                    help="Модули для инъекции (через запятую). Обычно 'qkv' или 'qkv,proj,fc1,fc2' для ViT")

    return ap.parse_args()


def safe_log10(x: float) -> float:
    if x is None or math.isnan(x) or x <= 0:
        return float("nan")
    return math.log10(x)


def compute_sam(acc_pct: float, energy_j: float, ab_values: list) -> Dict[str, float]:
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


def ensure_metrics_csv_header(ab_values: list):
    first = (not METRICS_CSV.exists()) or (METRICS_CSV.stat().st_size == 0)
    if first:
        with open(METRICS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            header = [
                "epoch",
                "train_time_s", "eval_time_s", "total_time_s",
                "train_energy_j", "eval_energy_j", "total_energy_j",
                "avg_power_w",
                "test_acc_pct"
            ]
            for a in ab_values:
                header.append(f"SAM_a{a}_b{a}")
            w.writerow(header)


def run_eval(model, loader, device, pwr, epoch_idx=-1):
    """
    Запускает валидацию на переданном loader.
    epoch_idx=-1 означает, что это Baseline (до обучения).
    Возвращает (accuracy_pct, avg_loss, metrics_dict)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0

    # Сброс метрик энергии только для фазы eval, если нужно,
    # но pwr.reset_epoch() обычно делается в начале эпохи.
    # Здесь мы просто логируем шаги.

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

            # Логируем шаг с epoch_idx
            # Если это baseline, можно передать epoch=0 или -1
            pwr.log_step("eval", epoch_idx, step, step_t, p_start, p_end)

    total_time = time.time() - start_t
    acc = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    return acc, avg_loss, total_time


def train():
    args = parse_args()
    set_seed(SEED)
    device = ensure_cuda()
    ckpt_mode = args.ckpt_mode

    # ---- GPU memory cap ----
    total_bytes = torch.cuda.get_device_properties(GPU_INDEX).total_memory
    cap_gb = max(0.1, MEMORY_CAPACITY_GB)
    cap_bytes = int(cap_gb * (1024 ** 3))

    try:
        frac = min(0.99, cap_bytes / total_bytes)
        torch.cuda.set_per_process_memory_fraction(frac, device=GPU_INDEX)
        print(f"[GPU MEM CAP] Limiting allocator to ~{cap_gb:.2f} GB ({frac * 100:.1f}%).")
    except AttributeError:
        pass  # Fallback logic omitted for brevity

    # SAM config
    ab_vals = [int(s.strip()) for s in args.sam_ab.split(",") if s.strip()]
    ab_vals = sorted(set([v for v in ab_vals if v >= 1]))
    ensure_metrics_csv_header(ab_vals)

    train_loader, test_loader = make_dataloaders(args.steps_per_epoch, args.batch_size, args.num_workers)

    # Строим модель
    use_static_ckpt = (ckpt_mode == "static")
    model = build_model(NUM_CLASSES,
                        ckpt=use_static_ckpt,
                        use_lora=args.use_lora,
                        lora_args=args).to(device)

    # Оптимизатор
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    cur_epoch = {"v": 0}
    cur_step = {"v": 0}
    epoch_ref = lambda: cur_epoch["v"]
    step_ref = lambda: cur_step["v"]

    base_model_for_hooks = get_base_model_from_peft(model)
    memlog = MemLogger(device, n_layers=len(base_model_for_hooks.blocks))
    pwr = GpuPowerMeter(device_index=GPU_INDEX)

    # Адаптивное чекпоинтирование
    if ckpt_mode == "adaptive":
        inject_dynamic_checkpointing(
            base_model_for_hooks,
            device=device,
            mem_cap_bytes=cap_bytes,
            step_ref=step_ref,
            memlog=memlog,
            epoch_ref=epoch_ref,
            pwr=pwr,
            threshold_ratio=0.55
        )

    handles = attach_mem_hooks(base_model_for_hooks, memlog, epoch_ref, step_ref)

    try:
        # ---------------------------------------------------------
        # 1) BASELINE EVALUATION (До тренировки)
        # ---------------------------------------------------------
        print(">>> Запуск Baseline оценки (Untrained Head)...")
        pwr.reset_epoch()  # Сброс счетчиков энергии для чистоты
        base_acc, base_loss, base_time = run_eval(model, test_loader, device, pwr, epoch_idx=0)
        print(f"[BASELINE] Acc: {base_acc:.2f}% | Loss: {base_loss:.4f} | Time: {base_time:.2f}s")
        print(">>> Обратите внимание: точность низкая, так как классификационная 'голова' инициализирована случайно.")

        # Можем записать baseline в CSV как эпоху 0
        with open(METRICS_CSV, "a", newline="") as f:
            # Заполняем нулями поля тренировки
            row = [0, 0.0, f"{base_time:.3f}", f"{base_time:.3f}", 0.0, 0.0, 0.0, 0.0, f"{base_acc:.2f}"]
            for _ in ab_vals: row.append("nan")
            csv.writer(f).writerow(row)
        # ---------------------------------------------------------

        global_step = 0
        for epoch in range(1, args.epochs + 1):
            cur_epoch["v"] = epoch
            memlog.reset_epoch_acc()
            pwr.reset_epoch()

            # --- TRAIN LOOP ---
            model.train()
            torch.cuda.reset_peak_memory_stats(device)
            loss_smooth = SmoothedValue(0.98)
            start_epoch_t = time.time()

            iterator = enumerate(train_loader, start=1)
            if not args.no_progress:
                iterator = tqdm(iterator, total=args.steps_per_epoch, ncols=120, leave=False, desc=f"Epoch {epoch}")

            for step, (x, y) in iterator:
                cur_step["v"] = step
                # ... (стандартный код шага обучения, без изменений) ...
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
                step_t = (time.time() - t0)
                p_end = pwr.sample_power_w()

                pwr.log_step("train", epoch, step, step_t, p_start, p_end)
                loss_smooth.update(loss.item())

                # Логирование в консоль (без изменений)
                if step % args.log_interval == 0:
                    alloc = bytes_to_mib(torch.cuda.memory_allocated(device))
                    peak = bytes_to_mib(torch.cuda.max_memory_allocated(device))
                    lr_curr = optimizer.param_groups[0]["lr"]
                    print(
                        f"[E{epoch:02d} S{step:05d}] loss={loss.item():.4f} sm={loss_smooth.value:.4f} lr={lr_curr:.2e} alloc={alloc:.0f}MiB")

                global_step += 1
                if step >= args.steps_per_epoch:
                    break

            dt_epoch_train = time.time() - start_epoch_t

            # --- EVAL LOOP ---
            acc, val_loss, dt_epoch_eval = run_eval(model, test_loader, device, pwr, epoch_idx=epoch)

            # --- LOGGING ---
            memlog.flush_epoch_avg(epoch)
            totals = pwr.epoch_totals()
            sam_vals = compute_sam(acc, totals["total_energy_j"], ab_vals)
            peak_epoch = bytes_to_mib(torch.cuda.max_memory_allocated(device))

            print(
                f"[Epoch {epoch}/{args.epochs}] "
                f"Acc={acc:.2f}% | TrainT={dt_epoch_train / 60:.1f}m | "
                f"EvalT={dt_epoch_eval:.1f}s | AvgW={totals['avg_power_w']:.1f} | "
                f"Energy={totals['total_energy_j']:.0f}J"
            )

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
                val = sam_vals[key]
                row.append(f"{val:.6f}" if not math.isnan(val) else "nan")

            with open(METRICS_CSV, "a", newline="") as f:
                csv.writer(f).writerow(row)

    finally:
        for h in handles: h.remove()
        memlog.close()
        pwr.close()

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

    # Используем model.blocks (предполагая, что сюда передали unwrap-нутую модель)
    for i, block in enumerate(model.blocks, start=1):
        h1 = block.register_forward_pre_hook(make_fwd_pre(i), with_kwargs=False)
        h2 = block.register_forward_hook(make_fwd_end(i))
        handles.extend([h1, h2])

        if have_bwd_pre:
            hb1 = block.register_full_backward_pre_hook(make_bwd_pre(i))
            hb2 = block.register_full_backward_hook(make_bwd_end(i))
            handles.extend([hb1, hb2])
        else:
            hb2 = block.register_full_backward_hook(make_bwd_end(i))
            handles.append(hb2)

    return handles


def inject_dynamic_checkpointing(model: nn.Module,
                                 device: torch.device,
                                 mem_cap_bytes: int,
                                 step_ref,
                                 memlog: MemLogger,
                                 epoch_ref,
                                 pwr=None,
                                 threshold_ratio: float = 0.8):
    # Работаем с unwrap-нутой моделью
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


if __name__ == "__main__":
    train()