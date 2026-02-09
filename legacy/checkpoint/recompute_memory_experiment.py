import os, gc, time, math, psutil
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import matplotlib.pyplot as plt


# ---------------------- Конфиг эксперимента ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Размеры подобраны так, чтобы на 8–12 ГБ GPU был заметный выигрыш,
# но при необходимости уменьшайте BATCH/HIDDEN/DEPTH.
BATCH = 64
SEQ_LEN = 1               # для MLP можно держать =1
HIDDEN = 4096             # 4096 даёт хороший след памяти, уменьшайте если OOM
DEPTH = 48                # кол-во линейных блоков (Linear+ReLU)
CHECKPOINT_CHUNKS = 12    # на сколько кусков резать при rematerialization

DTYPE = torch.float32  # чуток снижает память
TORCH_SDPA = False  # не нужно, оставлено на будущее

SAVE_PLOT = "ram_over_time.png"
SAVE_CSV  = "ram_over_time.csv"
# -----------------------------------------------------------------


@dataclass
class Trace:
    label: str
    times: List[float]
    mem_gb: List[float]


def now_s():
    return time.perf_counter()


def gpu_mem_gb():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024**3)
    # CPU fallback: RSS процесса
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


class DeepMLP(nn.Module):
    def __init__(self, width: int, depth: int):
        super().__init__()
        layers = []
        # Входной слой
        layers += [nn.Linear(width, width)]
        # Глубокая "труба"
        for _ in range(depth - 1):
            layers += [nn.ReLU(inplace=False), nn.Linear(width, width)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, use_ckpt: bool = False, chunks: int = 1):
        if use_ckpt:
            return checkpoint_sequential(self.net, chunks, x)
        else:
            return self.net(x)


def run_once(model: nn.Module, use_ckpt: bool, chunks: int, label: str) -> Trace:
    model.to(DEVICE).train()
    torch.manual_seed(0)

    # данные
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN, device=DEVICE, dtype=DTYPE, requires_grad=False)
    x = x.view(BATCH, HIDDEN)
    y = torch.randint(0, HIDDEN, (BATCH,), device=DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # логирование "по времени" через forward/backward хуки
    t0 = now_s()
    times, mems = [], []

    def log_point(tag: str):
        times.append(now_s() - t0)
        mems.append(gpu_mem_gb())

    # чистим счётчики CUDA
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # хуки на каждый модуль (Linear/ReLU)
    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.ReLU)):
            def fwd_hook(_m, _inp, _out):
                log_point("fwd")
            def bwd_hook(_m, _gin, _gout):
                log_point("bwd")
            handles.append(m.register_forward_hook(fwd_hook))
            # full_backward_hook, чтобы стабильно ловить во всех версиях
            handles.append(m.register_full_backward_hook(bwd_hook))

    # стартовое измерение
    log_point("start")

    # ---- forward
    out = model(x, use_ckpt=use_ckpt, chunks=chunks)
    # простая задача классификации: уменьшаем до logits
    logits = out
    # если HIDDEN маленький, можно сделать проекцию
    if logits.shape[-1] != HIDDEN:
        logits = F.linear(logits, torch.empty(HIDDEN, logits.shape[-1], device=DEVICE, dtype=DTYPE))
    loss = criterion(logits, y)

    log_point("after_forward")

    # ---- backward + step
    opt.zero_grad(set_to_none=True)
    loss.backward()
    log_point("after_backward")
    opt.step()
    log_point("after_step")

    # снимаем хуки
    for h in handles:
        h.remove()

    # небольшая уборка
    del x, y, out, logits, loss
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return Trace(label=label, times=times, mem_gb=mems)


def normalize_time(tr: Trace) -> Trace:
    # нормируем время в [0, 1] для сопоставимости двух прогонов
    if not tr.times:
        return tr
    t0, t1 = tr.times[0], tr.times[-1]
    span = max(1e-9, t1 - t0)
    times = [(t - t0) / span for t in tr.times]
    return Trace(label=tr.label, times=times, mem_gb=tr.mem_gb)


def main():
    print(f"Device: {DEVICE} | dtype: {DTYPE} | depth={DEPTH}, hidden={HIDDEN}, batch={BATCH}")
    model = DeepMLP(HIDDEN, DEPTH)

    # прогон 1: без checkpointing
    tr_keep = run_once(model, use_ckpt=False, chunks=1, label="Retain all activations")

    # прогон 2: с rematerialization
    tr_ckpt = run_once(model, use_ckpt=True, chunks=CHECKPOINT_CHUNKS, label="Rematerialize activations")

    tr_keep = normalize_time(tr_keep)
    tr_ckpt = normalize_time(tr_ckpt)

    # сохранить CSV
    import csv
    with open(SAVE_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "t_norm", "mem_gb"])
        for t, m in zip(tr_keep.times, tr_keep.mem_gb):
            w.writerow([tr_keep.label, t, m])
        for t, m in zip(tr_ckpt.times, tr_ckpt.mem_gb):
            w.writerow([tr_ckpt.label, t, m])
    print(f"Saved CSV -> {SAVE_CSV}")

    # построить график
    plt.figure(figsize=(4.4, 2.4), dpi=200)
    ax = plt.gca()
    ax.set_xlabel("Time")
    ax.set_ylabel("RAM used (GB)")
    ax.set_ylim(0, max(tr_keep.mem_gb + tr_ckpt.mem_gb) * 1.15)

    ax.plot(tr_keep.times, tr_keep.mem_gb, linewidth=2, label=tr_keep.label)
    ax.plot(tr_ckpt.times, tr_ckpt.mem_gb, linewidth=2, label=tr_ckpt.label)
    ax.fill_between(tr_ckpt.times, tr_ckpt.mem_gb, alpha=0.25, step="pre")

    # вертикальные пунктиры приблизительных границ backward
    # (по всплеску памяти после forward)
    try:
        # Heuristic: момент после_forward — почти граница
        t_keep_cut = min(tr_keep.times[-2], 0.8)
        t_ckpt_cut = min(tr_ckpt.times[-2], 0.8)
        ax.axvline(t_keep_cut, linestyle="--", linewidth=1)
        ax.axvline(t_ckpt_cut, linestyle="--", linewidth=1)
    except Exception:
        pass

    ax.legend(frameon=False, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(SAVE_PLOT)
    print(f"Saved plot -> {SAVE_PLOT}")


if __name__ == "__main__":
    main()
