
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ViT-S/16 on CIFAR-100 while measuring GPU memory per block.
Supports activation checkpointing via timm's set_grad_checkpointing(True).

Outputs (in ./mem_logs by default):
  - memlog_raw.csv.gz       : per-step raw samples (epoch, step, phase, layer, mem_mib)
  - memlog_epoch_avg.csv    : per-epoch avg per (phase, layer), for quick plotting

Examples:
  python train_vit_cifar100_memlog_ckpt.py --epochs 2 --batch-size 128
  python train_vit_cifar100_memlog_ckpt.py --epochs 2 --batch-size 128 --grad-checkpointing
  python train_vit_cifar100_memlog_ckpt.py --epochs 2 --batch-size 128 --amp
"""
import os, csv, gzip, math, time, argparse
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# pip install timm tqdm
import timm
from tqdm import tqdm

# -------------------------- Utils & Logging ----------------------------------

def mib(x: int) -> float:
    return x / (1024 * 1024)

class MemLogger:
    """Log memory samples at fwd/bwd hooks per layer; write raw + epoch averages."""
    def __init__(self, out_dir: Path, world_rank: int = 0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.out_dir / "memlog_raw.csv.gz"
        self.epoch_avg_path = self.out_dir / "memlog_epoch_avg.csv"
        self.world_rank = world_rank

        # in-memory buffers for current epoch stats: sums & counts
        self._acc: Dict[Tuple[str,int], Tuple[float,int]] = {}

        # set header for raw file
        with gzip.open(self.raw_path, "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","step","phase","layer","mem_mib"])

        # if epoch avg doesn't exist, create with header
        if not self.epoch_avg_path.exists():
            with open(self.epoch_avg_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch","phase","layer","x_label","mem_mib"])

    def sample(self, epoch:int, step:int, phase:str, layer_idx:int):
        if not torch.cuda.is_available():
            return
        # use allocated, as it correlates well with model live tensors
        mem = mib(torch.cuda.memory_allocated())
        # write raw row (append)
        with gzip.open(self.raw_path, "at", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, step, phase, layer_idx, f"{mem:.3f}"])
        # accumulate for epoch avg
        key = (phase, layer_idx)
        if key not in self._acc:
            self._acc[key] = (mem, 1)
        else:
            s, c = self._acc[key]
            self._acc[key] = (s + mem, c + 1)

    def end_epoch(self, epoch:int):
        # dump epoch averages
        rows = []
        for (phase, layer_idx), (s, c) in self._acc.items():
            avg = s / max(1, c)
            x_label = f"{phase}-L{layer_idx}"
            rows.append((phase, layer_idx, x_label, avg))
        # canonical order: fwd L1..Ln, bwd Ln..L1
        rows_fwd = sorted([r for r in rows if r[0]=="fwd"], key=lambda r: r[1])
        rows_bwd = sorted([r for r in rows if r[0]=="bwd"], key=lambda r: -r[1])
        rows = rows_fwd + rows_bwd

        with open(self.epoch_avg_path, "a", newline="") as f:
            w = csv.writer(f)
            for phase, layer_idx, x_label, avg in rows:
                w.writerow([epoch, phase, layer_idx, x_label, f"{avg:.3f}"])

        self._acc.clear()

# -------------------------- Model --------------------------------------------

def build_vit_s(num_classes:int, img_size:int=224, ckpt:bool=False):
    # vit_small_patch16_224 has 12 blocks
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        img_size=img_size,
    )
    if ckpt and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(enable=True)
    return model

def register_mem_hooks(model: nn.Module, memlog: MemLogger):
    """Attach hooks to encoder blocks. Returns a list of handles to keep alive."""
    handles = []
    # timm's VisionTransformer has model.blocks (list of nn.Module)
    assert hasattr(model, "blocks"), "Unexpected ViT structure (no .blocks)"
    blocks: List[nn.Module] = list(model.blocks)

    def make_fwd_hook(idx):
        def hook(module, inputs, output):
            memlog.sample(memlog._cur_epoch, memlog._cur_step, "fwd", idx)
        return hook

    def make_bwd_hook(idx):
        def hook(module, grad_input, grad_output):
            memlog.sample(memlog._cur_epoch, memlog._cur_step, "bwd", idx)
        return hook

    for i, blk in enumerate(blocks, start=1):
        handles.append(blk.register_forward_hook(make_fwd_hook(i)))
        handles.append(blk.register_full_backward_hook(make_bwd_hook(i)))
    return handles

# -------------------------- Data ---------------------------------------------

def make_dataloaders(batch_size:int, num_workers:int):
    mean=(0.5071, 0.4867, 0.4408); std=(0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

# -------------------------- Train / Eval --------------------------------------

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.size(0)))
    return res

def train_one_epoch(model, loader, optimizer, scaler, device, memlog: MemLogger, epoch:int, log_interval:int=100):
    model.train()
    criterion = nn.CrossEntropyLoss()
    # annotate current epoch for hooks
    memlog._cur_epoch = epoch
    pbar = tqdm(enumerate(loader, start=1), total=len(loader), leave=False, desc=f"epoch {epoch}")
    for step, (images, targets) in pbar:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        memlog._cur_step = step

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if step % log_interval == 0:
            top1 = accuracy(logits, targets, topk=(1,))[0].item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc1=f"{top1:.1f}%")

    memlog.end_epoch(epoch)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        pred = logits.argmax(1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total

# -------------------------- Main ----------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision")
    ap.add_argument("--grad-checkpointing", action="store_true", help="Enable activation checkpointing in ViT")
    ap.add_argument("--mem-dir", type=Path, default=Path("./mem_logs"))
    ap.add_argument("--log-interval", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader = make_dataloaders(args.batch_size, args.num_workers)
    model = build_vit_s(num_classes=100, ckpt=args.grad-checkpointing).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    memlog = MemLogger(args.mem_dir)

    # register hooks after model is on device
    handles = register_mem_hooks(model, memlog)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scaler, device, memlog, epoch, args.log_interval)
        acc1 = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc1)
        print(f"Epoch {epoch}: val@1 = {acc1:.2f}%  (best {best_acc:.2f}%)")

    # keep handles alive till the end
    for h in handles:
        h.remove()

if __name__ == "__main__":
    main()
