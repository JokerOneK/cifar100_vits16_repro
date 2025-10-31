import os
import math
import time
import gzip
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime  # [TIME-LAYER]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms

# pip install timm tqdm
import timm
from tqdm import tqdm  # [NEW]

# -------------------------------
# Константы по умолчанию (можно переопределить через CLI)
# -------------------------------
EPOCHS = 2
STEPS_PER_EPOCH = 782
BATCH_SIZE = 64
NUM_WORKERS = 4
MODEL_NAME = "vit_small_patch16_224"
NUM_CLASSES = 100
SEED = 42
OUTDIR = Path("./mem_logs")
OUTDIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = False

RAW_LOG_GZ = OUTDIR / "memlog_raw.csv.gz"
EPOCH_AVG_CSV = OUTDIR / "memlog_epoch_avg.csv"
PLOT_PNG = OUTDIR / "memplot_epochs.png"


# [TIME-LAYER] новые файлы логов
LAYER_TIMES_GZ = OUTDIR / "layer_times.csv.gz"
LAYER_TIME_EPOCH_AVG_CSV = OUTDIR / "layer_time_epoch_avg.csv"


def iso_now():  # [TIME-LAYER]
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
        raise RuntimeError("CUDA GPU не обнаружена. Запустите на машине с NVIDIA GPU и установленными CUDA-демонами.")
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
    test_ds  = datasets.CIFAR100(root="./data", train=False, transform=test_tf, download=True)

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

def build_model(num_classes: int, ckpt: bool = False):
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    assert hasattr(model, "blocks") and len(model.blocks) == 12, "Ожидались 12 блоков в ViT-S/16"

    if ckpt and hasattr(model, "set_grad_checkpointing"):

        model.set_grad_checkpointing(enable=True)
        print(f'Чекпоинт включен')
    return model

class MemLogger:
    def __init__(self, device, n_layers: int):
        self.device = device
        self.n_layers = n_layers

        # ----- как было -----
        self.raw_file = gzip.open(RAW_LOG_GZ, "at", newline="")
        self.raw_writer = csv.writer(self.raw_file)
        if RAW_LOG_GZ.stat().st_size == 0:
            self.raw_writer.writerow(["epoch", "step", "phase", "layer", "mem_mib"])
        self.epoch_acc: Dict[Tuple[str,int], Tuple[float,int]] = {}

        # ----- [TIME] пошаговые метрики шага (если не добавлял раньше) -----
        # (можно оставить как у тебя в прошлом патче)

        # ----- [TIME-LAYER] покадровые тайминги блоков -----
        self.time_file = gzip.open(LAYER_TIMES_GZ, "at", newline="")
        self.time_writer = csv.writer(self.time_file)
        if LAYER_TIMES_GZ.stat().st_size == 0:
            self.time_writer.writerow(["ts", "epoch", "step", "phase", "layer", "ms"])

        # аккумулируем средние по эпохе времени на слой/фазу
        # ключ: (phase, layer_idx) -> (total_ms, count)
        self.epoch_time_acc: Dict[Tuple[str,int], Tuple[float,int]] = {}

        # [TIME-LAYER] заголовок для сводки по эпохе
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

    # [TIME-LAYER] логируем тайминг одного слоя (фаза fwd/bwd)
    def log_layer_time(self, epoch: int, step: int, phase: str, layer_idx: int, ms: float):
        self.time_writer.writerow([iso_now(), epoch, step, phase, layer_idx, f"{ms:.3f}"])
        key = (phase, layer_idx)
        total, cnt = self.epoch_time_acc.get(key, (0.0, 0))
        self.epoch_time_acc[key] = (total + ms, cnt + 1)

    def reset_epoch_acc(self):
        self.epoch_acc.clear()
        self.epoch_time_acc.clear()  # [TIME-LAYER]

    def flush_epoch_avg(self, epoch: int):
        # память (как было)
        first_write = not EPOCH_AVG_CSV.exists() or EPOCH_AVG_CSV.stat().st_size == 0
        with open(EPOCH_AVG_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if first_write:
                w.writerow(["epoch", "x_label", "phase", "layer", "mem_mib"])
            for i in range(1, self.n_layers + 1):
                total, cnt = self.epoch_acc.get(("fwd", i), (0.0, 1))
                w.writerow([epoch, f"fwd-L{i}", "fwd", i, f"{(total/cnt):.3f}"])
            for i in range(self.n_layers, 0, -1):
                total, cnt = self.epoch_acc.get(("bwd", i), (0.0, 1))
                w.writerow([epoch, f"bwd-L{i}", "bwd", i, f"{(total/cnt):.3f}"])

        # [TIME-LAYER] средние времена по слоям за эпоху
        with open(LAYER_TIME_EPOCH_AVG_CSV, "a", newline="") as f:
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
        try: self.time_file.close()         # [TIME-LAYER]
        except Exception: pass
        # если добавлял step/epoch summary — их тоже закрыть


def attach_mem_hooks(model: nn.Module, memlog: MemLogger, epoch_ref, step_ref):
    """
    Регистрирует:
      - forward_pre_hook: лог памяти (как у тебя) + старт fwd-таймера
      - forward_hook: лог конца fwd-таймера
      - full_backward_pre_hook: старт bwd-таймера
      - full_backward_hook: конец bwd-таймера
    """
    handles = []

    # события для измерения времени на GPU по слоям
    fwd_start_events: Dict[int, torch.cuda.Event] = {}
    bwd_start_events: Dict[int, torch.cuda.Event] = {}

    def make_fwd_pre(layer_idx):
        def _hook(module, inp):
            # память как раньше
            memlog.log_now(epoch_ref(), step_ref(), "fwd", layer_idx)
            # старт fwd-таймера
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(torch.cuda.current_stream())
            fwd_start_events[layer_idx] = ev
        return _hook

    def make_fwd_end(layer_idx):
        def _hook(module, inp, out):
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream())
            # синхронизация только для корректного elapsed_time (локально)
            end.synchronize()
            start = fwd_start_events.pop(layer_idx, None)
            if start is not None:
                ms = start.elapsed_time(end)  # CUDA time (ms)
                memlog.log_layer_time(epoch_ref(), step_ref(), "fwd", layer_idx, ms)
        return _hook

    # backward: нужен PyTorch >= 1.13 для pre-hook
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
            if start is not None:
                ms = start.elapsed_time(end)
            else:
                # fallback: если нет pre-hook, логируем 0 — или можно пропустить запись
                ms = 0.0
            # логируем память на bwd, как у тебя
            memlog.log_now(epoch_ref(), step_ref(), "bwd", layer_idx)
            memlog.log_layer_time(epoch_ref(), step_ref(), "bwd", layer_idx, ms)
        return _hook

    for i, block in enumerate(model.blocks, start=1):
        # FWD
        h1 = block.register_forward_pre_hook(make_fwd_pre(i), with_kwargs=False)
        h2 = block.register_forward_hook(make_fwd_end(i))
        handles.extend([h1, h2])

        # BWD
        if have_bwd_pre:
            hb1 = block.register_full_backward_pre_hook(make_bwd_pre(i))
            hb2 = block.register_full_backward_hook(make_bwd_end(i))
            handles.extend([hb1, hb2])
        else:
            # Если нет pre-hook, хотя бы конец (время будет 0.0 по fallback)
            hb2 = block.register_full_backward_hook(make_bwd_end(i))
            handles.append(hb2)

    return handles


# [NEW] — простая метрика сглаженного среднего
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
    ap.add_argument("--log-interval", type=int, default=200, help="Каждые N шагов печатать детальный лог")
    ap.add_argument("--no-progress", action="store_true", help="Отключить tqdm прогресс-бар")
    ap.add_argument("--amp", action="store_true", help="Включить torch.cuda.amp.autocast()")
    return ap.parse_args()

def train():
    args = parse_args()
    set_seed(SEED)
    device = ensure_cuda()
    ckpt = CHECKPOINT

    train_loader, test_loader = make_dataloaders(args.steps_per_epoch, args.batch_size, args.num_workers)
    model = build_model(NUM_CLASSES, ckpt=ckpt).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)  # [NEW]

    cur_epoch = {"v": 0}
    cur_step  = {"v": 0}
    epoch_ref = lambda: cur_epoch["v"]
    step_ref  = lambda: cur_step["v"]

    memlog = MemLogger(device, n_layers=len(model.blocks))
    handles = attach_mem_hooks(model, memlog, epoch_ref, step_ref)

    try:
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            cur_epoch["v"] = epoch
            memlog.reset_epoch_acc()
            model.train()
            torch.cuda.reset_peak_memory_stats(device)

            loss_smooth = SmoothedValue(0.98)  # [NEW]
            start_epoch_t = time.time()
            num_images = 0

            iterator = enumerate(train_loader, start=1)
            if not args.no_progress:
                iterator = tqdm(iterator, total=args.steps_per_epoch, ncols=120, leave=False,
                                desc=f"Epoch {epoch}/{args.epochs}")

            for step, (x, y) in iterator:
                cur_step["v"] = step
                torch.cuda.synchronize()
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
                num_images += x.size(0)
                ips = x.size(0) / step_t

                loss_smooth.update(loss.item())

                alloc = bytes_to_mib(torch.cuda.memory_allocated(device))
                peak  = bytes_to_mib(torch.cuda.max_memory_allocated(device))
                lr = next(iter(optimizer.param_groups))["lr"]

                # [NEW] tqdm статус
                if not args.no_progress:
                    iterator.set_postfix({
                        "step": f"{step}/{args.steps_per_epoch}",
                        "loss": f"{loss_smooth.value:.4f}",
                        "lr": f"{lr:.2e}",
                        "ms": f"{step_t*1000:.0f}",
                        "img/s": f"{ips:.0f}",
                        "alloc": f"{alloc:.0f}MiB",
                        "peak": f"{peak:.0f}MiB"
                    })

                # [NEW] периодический подробный принт (полезно, если пишете лог в файл)
                if step % args.log_interval == 0 or step == 1:
                    print(
                        f"[E{epoch:02d}/{args.epochs} S{step:05d}/{args.steps_per_epoch}] "
                        f"loss={loss.item():.4f} loss_smooth={loss_smooth.value:.4f} "
                        f"lr={lr:.2e} step_ms={step_t*1000:.0f} img_s={ips:.0f} "
                        f"alloc={alloc:.1f}MiB peak={peak:.1f}MiB"
                    )

                global_step += 1
                if step >= args.steps_per_epoch:
                    break

            # итог по эпохе
            dt_epoch = time.time() - start_epoch_t
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            acc = 100.0 * correct / total
            peak_epoch = bytes_to_mib(torch.cuda.max_memory_allocated(device))

            memlog.flush_epoch_avg(epoch)

            print(
                f"[Epoch {epoch}/{args.epochs}] "
                f"test_acc={acc:.2f}% | time={dt_epoch/60:.2f} min | "
                f"avg_ips={(num_images/dt_epoch):.0f} img/s | peak_mem={peak_epoch:.1f} MiB"
            )

    finally:
        for h in handles:
            h.remove()
        memlog.close()

if __name__ == "__main__":
    """
    Примеры запуска:
      # стандартный прогресс + принты каждые 200 шагов
      python train_vit_cifar100_memlog.py

      # выключить прогресс-бар, оставив только принты раз в 500 шагов
      python train_vit_cifar100_memlog.py --no-progress --log-interval 500

      # включить AMP для ускорения и меньшей памяти
      python train_vit_cifar100_memlog.py --amp
    """
    train()
