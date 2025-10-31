import os
import math
import time
import gzip
import csv
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms

# pip install timm
import timm

# -------------------------------
# Константы из запроса
# -------------------------------
EPOCHS = 10
STEPS_PER_EPOCH = 15640                      # как вы и задали
BATCH_SIZE = 64                               # можно менять; на шаги/эпоху не влияет
NUM_WORKERS = 4
MODEL_NAME = "vit_small_patch16_224"          # ViT-S/16 (12 encoder blocks) в timm
NUM_CLASSES = 100                             # CIFAR-100
SEED = 42
OUTDIR = Path("./mem_logs")
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_LOG_GZ = OUTDIR / "memlog_raw.csv.gz"     # подробный лог по каждому шагу
EPOCH_AVG_CSV = OUTDIR / "memlog_epoch_avg.csv"  # среднее по эпохам (точки графика)
PLOT_PNG = OUTDIR / "memplot_epochs.png"

# -------------------------------
# Утилиты
# -------------------------------
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

# -------------------------------
# Датасет: CIFAR-100 -> resize до 224
# -------------------------------
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

    # Нам нужны фиксированные steps/epoch — используем семплер с заменой:
    # Общее кол-во семплов за эпоху = steps_per_epoch * batch_size
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

# -------------------------------
# Модель
# -------------------------------
def build_model(num_classes: int):
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    # На timm vit_small_patch16_224: model.blocks — это список из 12 encoder blocks
    assert hasattr(model, "blocks") and len(model.blocks) == 12, "Ожидались 12 блоков в ViT-S/16"
    return model

# -------------------------------
# Хуки для логирования памяти
# -------------------------------
class MemLogger:
    """
    Логирует GPU память на входе каждого блока (forward_pre) и во время backward для этого блока.
    Сохраняет в gzip-CSV строками: epoch, step, phase, layer, mem_mib
    Дополнительно копит сумму/счётчик для усреднения по эпохе.
    """
    def __init__(self, device, n_layers: int):
        self.device = device
        self.n_layers = n_layers
        self.raw_file = gzip.open(RAW_LOG_GZ, "at", newline="")
        self.raw_writer = csv.writer(self.raw_file)
        if RAW_LOG_GZ.stat().st_size == 0:
            self.raw_writer.writerow(["epoch", "step", "phase", "layer", "mem_mib"])  # header

        # Для усреднения по эпохам:
        # ключ = ("fwd", i) или ("bwd", i), значения: суммарная память и количество шагов
        self.epoch_acc: Dict[Tuple[str,int], Tuple[float,int]] = {}

    @torch.no_grad()
    def log_now(self, epoch: int, step: int, phase: str, layer_idx: int):
        mem = torch.cuda.memory_allocated(self.device)
        self.raw_writer.writerow([epoch, step, phase, layer_idx, f"{bytes_to_mib(mem):.3f}"])

        key = (phase, layer_idx)
        total, cnt = self.epoch_acc.get(key, (0.0, 0))
        self.epoch_acc[key] = (total + bytes_to_mib(mem), cnt + 1)

    def reset_epoch_acc(self):
        self.epoch_acc.clear()

    def flush_epoch_avg(self, epoch: int):
        # Записываем усреднённые по эпохе значения в удобном порядке оси X:
        # fwd-L1..L12, bwd-L12..L1
        with open(EPOCH_AVG_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if not EPOCH_AVG_CSV.exists() or EPOCH_AVG_CSV.stat().st_size == 0:
                w.writerow(["epoch", "x_label", "phase", "layer", "mem_mib"])

            # forward L1..L12
            for i in range(1, self.n_layers + 1):
                total, cnt = self.epoch_acc.get(("fwd", i), (0.0, 1))
                w.writerow([epoch, f"fwd-L{i}", "fwd", i, f"{(total/cnt):.3f}"])
            # backward L12..L1
            for i in range(self.n_layers, 0, -1):
                total, cnt = self.epoch_acc.get(("bwd", i), (0.0, 1))
                w.writerow([epoch, f"bwd-L{i}", "bwd", i, f"{(total/cnt):.3f}"])

    def close(self):
        try:
            self.raw_file.close()
        except Exception:
            pass

def attach_mem_hooks(model: nn.Module, memlog: MemLogger, epoch_ref, step_ref):
    """
    Вешаем forward_pre и full_backward хуки на каждый encoder block.
    epoch_ref, step_ref — лямбды, возвращающие текущий номер эпохи и шага.
    Возвращает список хэндлов, чтобы их потом снять.
    """
    handles = []

    def make_fwd_pre(layer_idx):
        def _hook(module, inp):
            memlog.log_now(epoch_ref(), step_ref(), "fwd", layer_idx)
        return _hook

    def make_bwd(layer_idx):
        def _hook(module, grad_input, grad_output):
            # backward идёт в порядке L12..L1
            memlog.log_now(epoch_ref(), step_ref(), "bwd", layer_idx)
        return _hook

    for i, block in enumerate(model.blocks, start=1):
        h1 = block.register_forward_pre_hook(make_fwd_pre(i), with_kwargs=False)
        h2 = block.register_full_backward_hook(make_bwd(i))
        handles.extend([h1, h2])

    return handles

# -------------------------------
# Обучение
# -------------------------------
def train():
    set_seed(SEED)
    device = ensure_cuda()

    train_loader, test_loader = make_dataloaders(STEPS_PER_EPOCH, BATCH_SIZE,)
    model = build_model(NUM_CLASSES).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()

    # Ссылки на текущие epoch/step для хуков
    cur_epoch = {"v": 0}
    cur_step  = {"v": 0}
    epoch_ref = lambda: cur_epoch["v"]
    step_ref  = lambda: cur_step["v"]

    memlog = MemLogger(device, n_layers=len(model.blocks))
    handles = attach_mem_hooks(model, memlog, epoch_ref, step_ref)

    try:
        global_step = 0
        for epoch in range(1, EPOCHS + 1):
            cur_epoch["v"] = epoch
            memlog.reset_epoch_acc()

            model.train()
            torch.cuda.reset_peak_memory_stats(device)

            # Цикл по фиксированному train_loader (с RandomSampler replacement уже настроен)
            for step, (x, y) in enumerate(train_loader, start=1):
                cur_step["v"] = step

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = model(x)                     # forward: тут сработают fwd-хуки по слоям
                loss = criterion(out, y)
                loss.backward()                    # backward: тут сработают bwd-хуки по слоям
                optimizer.step()

                global_step += 1
                if step >= STEPS_PER_EPOCH:
                    break

            # По завершении эпохи — агрегируем средние и добавляем в CSV для построения графика
            memlog.flush_epoch_avg(epoch)

            # Небольшая валидация (не обязательно, но полезно)
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
            print(f"[Epoch {epoch}/{EPOCHS}] test acc: {acc:.2f}%  | peak_mem: {bytes_to_mib(torch.cuda.max_memory_allocated(device)):.1f} MiB")

    finally:
        # Снимаем хуки и закрываем файлы
        for h in handles:
            h.remove()
        memlog.close()

if __name__ == "__main__":
    """
    Запуск:
      pip install torch torchvision timm matplotlib
      python train_vit_cifar100_memlog.py
    Логи появятся в ./mem_logs/
    """
    train()
