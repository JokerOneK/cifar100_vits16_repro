"""
DINOv2 k-NN & Linear Probe on CIFAR-100 / DTD
Adapted from dinov2.ipynb for local GPU (RTX 2060 Super 8GB).

Changes vs Kaggle notebook:
  - Paths: local instead of /kaggle/working/
  - Batch size: 32 (safe for 8 GB VRAM)
  - num_workers: 0 (Windows)
  - No pip install cell

Install:
  pip install transformers pynvml pandas
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:128,garbage_collection_threshold:0.8"
)

import math, time, gzip, csv, random, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from PIL import Image
from transformers import Dinov2Model, AutoImageProcessor
import torchvision

print(f"torch={torch.__version__}  torchvision={torchvision.__version__}  "
      f"CUDA={torch.version.cuda}")

# ========================= CONFIG =========================
MODEL_SIZE      = "large"       # "small" (22M) | "base" (86M) | "large" (300M) | "giant" (1.1B)
DATASETS        = ["cifar100", "dtd"]
EVAL_BATCH_SIZE = 32            # safe for 8 GB VRAM
KNN_K           = 20
LINEAR_EPOCHS   = 10
LINEAR_LR       = 1e-3
SEED            = 42
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR     = SCRIPT_DIR / "dinov2_results"
DATA_ROOT  = SCRIPT_DIR / "data"
GPU_INDEX  = 0

DINOV2_MODELS = {
    "small":  {"id": "facebook/dinov2-small",  "params_m": 22,   "dim": 384},
    "base":   {"id": "facebook/dinov2-base",   "params_m": 86,   "dim": 768},
    "large":  {"id": "facebook/dinov2-large",  "params_m": 300,  "dim": 1024},
    "giant":  {"id": "facebook/dinov2-giant",  "params_m": 1100, "dim": 1536},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram_gb:.1f} GB")


# ========================= HELPERS =========================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def iso_now(): return datetime.now().isoformat(timespec="seconds")
def safe_log10(x):
    if x is None or math.isnan(x) or x <= 0: return float("nan")
    return math.log10(x)

def _maybe_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ========================= DATASET =========================
class DinoDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        if hasattr(self.base, 'transform'): self.base.transform = None

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if isinstance(img, torch.Tensor): img = transforms.ToPILImage()(img)
        elif not isinstance(img, Image.Image): img = Image.fromarray(np.array(img))
        if img.mode != "RGB": img = img.convert("RGB")
        return img, label


def dino_collate(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def make_loaders(dataset_name):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if dataset_name == "cifar100":
        train_ds = datasets.CIFAR100(root=str(DATA_ROOT), train=True,
                                     transform=None, download=True)
        test_ds  = datasets.CIFAR100(root=str(DATA_ROOT), train=False,
                                     transform=None, download=True)
        num_classes = 100
    elif dataset_name == "dtd":
        train_ds = datasets.DTD(root=str(DATA_ROOT), split="train",
                                transform=None, download=True)
        test_ds  = datasets.DTD(root=str(DATA_ROOT), split="test",
                                transform=None, download=True)
        num_classes = 47
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    train_loader = DataLoader(DinoDataset(train_ds), batch_size=EVAL_BATCH_SIZE,
                              shuffle=False, num_workers=0,  # Windows
                              collate_fn=dino_collate,
                              pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(DinoDataset(test_ds), batch_size=EVAL_BATCH_SIZE,
                              shuffle=False, num_workers=0,  # Windows
                              collate_fn=dino_collate,
                              pin_memory=torch.cuda.is_available())
    return train_loader, test_loader, num_classes


# ========================= POWER =========================
class GpuPowerMeter:
    def __init__(self, device_index, step_energy_path):
        self.available = False; self.handle = None
        self.device_index = device_index
        try:
            import pynvml
            self.nvml = pynvml; self.nvml.nvmlInit()
            self.handle = self.nvml.nvmlDeviceGetHandleByIndex(device_index)
            _ = self.nvml.nvmlDeviceGetPowerUsage(self.handle)
            self.available = True
        except Exception: self.nvml = None
        self.reset_epoch()

        write_header = (not step_energy_path.exists()) or (step_energy_path.stat().st_size == 0)
        self._step_file = gzip.open(step_energy_path, "at", newline="")
        self._step_writer = csv.writer(self._step_file)
        if write_header:
            self._step_writer.writerow(["ts","epoch","step","phase","step_ms",
                                        "p_start_w","p_end_w","p_avg_w","energy_j"])

    def close(self):
        try:
            if self.available and self.nvml: self.nvml.nvmlShutdown()
        except: pass
        try: self._step_file.close()
        except: pass

    def sample_power_w(self):
        if not self.available: return float("nan")
        try: return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except: return float("nan")

    def reset_epoch(self):
        self.train_energy_j = 0.0; self.eval_energy_j = 0.0
        self.train_time_s = 0.0;   self.eval_time_s = 0.0

    def log_step(self, phase, epoch, step, step_time_s, p_start, p_end):
        p_avg = (p_start+p_end)/2.0 if (not math.isnan(p_start) and not math.isnan(p_end)) else float("nan")
        e = p_avg * step_time_s if not math.isnan(p_avg) else float("nan")
        if phase.startswith("train"):
            self.train_time_s += step_time_s
            if not math.isnan(p_avg) and not math.isnan(self.train_energy_j): self.train_energy_j += e
            elif math.isnan(p_avg): self.train_energy_j = float("nan")
        else:
            self.eval_time_s += step_time_s
            if not math.isnan(p_avg) and not math.isnan(self.eval_energy_j): self.eval_energy_j += e
            elif math.isnan(p_avg): self.eval_energy_j = float("nan")
        self._step_writer.writerow([
            iso_now(), epoch, step, phase, f"{step_time_s*1000:.3f}",
            f"{p_start:.3f}", f"{p_end:.3f}",
            f"{p_avg:.3f}" if not math.isnan(p_avg) else "nan",
            f"{e:.6f}" if not math.isnan(e) else "nan"
        ])

    def epoch_totals(self):
        te = self.train_energy_j if not math.isnan(self.train_energy_j) else 0.0
        ee = self.eval_energy_j  if not math.isnan(self.eval_energy_j)  else 0.0
        total_e = te + ee
        if math.isnan(self.train_energy_j) and math.isnan(self.eval_energy_j): total_e = float("nan")
        total_t = self.train_time_s + self.eval_time_s
        avg_p = (total_e / total_t) if (not math.isnan(total_e) and total_t > 0) else float("nan")
        return dict(train_energy_j=self.train_energy_j, eval_energy_j=self.eval_energy_j,
                    total_energy_j=total_e, train_time_s=self.train_time_s,
                    eval_time_s=self.eval_time_s, total_time_s=total_t, avg_power_w=avg_p)


def compute_sam(acc_pct, energy_j, ab_values):
    acc = acc_pct / 100.0; logE = safe_log10(energy_j)
    return {f"SAM_a{a}_b{a}": (acc**a)/(logE**a) if (not math.isnan(logE) and acc>0) else float("nan")
            for a in ab_values}


def ensure_metrics_csv_header(ab_values, path):
    if (not path.exists()) or (path.stat().st_size == 0):
        with open(path, "a", newline="") as f:
            h = ["epoch","method","train_time_s","eval_time_s","total_time_s",
                 "train_energy_j","eval_energy_j","total_energy_j","avg_power_w","test_acc_pct"]
            for a in ab_values: h.append(f"SAM_a{a}_b{a}")
            csv.writer(f).writerow(h)


# ========================= MODEL =========================
def build_dinov2(model_size):
    info = DINOV2_MODELS[model_size]
    print(f"[DINOv2] Loading {info['id']} (~{info['params_m']}M params, dim={info['dim']})...")
    processor = AutoImageProcessor.from_pretrained(info["id"])
    model = Dinov2Model.from_pretrained(info["id"], torch_dtype=torch.float16).to(device)
    model.eval()
    print(f"[DINOv2] Total params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, processor


@torch.no_grad()
def extract_features(model, processor, loader, pwr, phase="eval", epoch_idx=0):
    """Extract CLS token features from frozen DINOv2."""
    model.eval()
    all_feats, all_labels = [], []

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"Extracting ({phase})", leave=True)
    for step, (images, labels) in pbar:
        _maybe_sync()
        p_start = pwr.sample_power_w()
        t0 = time.time()

        inp = processor(images=images, return_tensors="pt").to(device)
        pixel_values = inp["pixel_values"].to(dtype=model.dtype)
        out = model(pixel_values=pixel_values)
        cls_feats = out.last_hidden_state[:, 0].float()
        cls_feats = F.normalize(cls_feats, dim=-1)

        all_feats.append(cls_feats.cpu())
        all_labels.append(labels)

        _maybe_sync()
        step_t = time.time() - t0
        pwr.log_step(phase, epoch_idx, step, step_t, p_start, pwr.sample_power_w())

    return torch.cat(all_feats), torch.cat(all_labels)


# ========================= k-NN =========================
def knn_classify(train_feats, train_labels, test_feats, test_labels, k=20):
    num_classes = train_labels.max().item() + 1
    train_f = train_feats.to(device)
    test_f  = test_feats.to(device)
    train_l = train_labels.to(device)

    correct = 0
    total = len(test_labels)
    chunk_size = 256

    for i in tqdm(range(0, total, chunk_size), desc=f"k-NN (k={k})"):
        chunk = test_f[i:i+chunk_size]
        sims = chunk @ train_f.T
        topk_sims, topk_idx = sims.topk(k, dim=-1)
        topk_labels = train_l[topk_idx]

        one_hot = F.one_hot(topk_labels, num_classes).float()
        weighted = (one_hot * topk_sims.unsqueeze(-1)).sum(dim=1)
        preds = weighted.argmax(dim=-1).cpu()

        correct += (preds == test_labels[i:i+chunk_size]).sum().item()

    acc = 100.0 * correct / total
    print(f"  k-NN Accuracy (k={k}): {acc:.2f}%  ({correct}/{total})")
    return acc


# ========================= LINEAR PROBE =========================
def linear_probe(train_feats, train_labels, test_feats, test_labels,
                 num_classes, pwr, epochs=10, lr=1e-3, batch_size=256):
    feat_dim = train_feats.shape[1]
    head = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    train_f = train_feats.to(device)
    train_l = train_labels.to(device)
    test_f  = test_feats.to(device)
    test_l  = test_labels.to(device)

    n_train = len(train_labels)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            _maybe_sync()
            p_start = pwr.sample_power_w()
            t0 = time.time()

            idx = perm[i:i+batch_size]
            logits = head(train_f[idx])
            loss = criterion(logits, train_l[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            _maybe_sync()
            pwr.log_step("train", epoch, i // batch_size, time.time() - t0,
                         p_start, pwr.sample_power_w())

        head.eval()
        with torch.no_grad():
            _maybe_sync()
            p_start = pwr.sample_power_w()
            t0 = time.time()

            logits = head(test_f)
            preds = logits.argmax(dim=-1)
            acc = 100.0 * (preds == test_l).float().mean().item()

            _maybe_sync()
            pwr.log_step("eval", epoch, 0, time.time() - t0,
                         p_start, pwr.sample_power_w())

        best_acc = max(best_acc, acc)
        print(f"  Epoch {epoch}/{epochs}: loss={epoch_loss/n_batches:.4f}  "
              f"acc={acc:.2f}%  best={best_acc:.2f}%")

    return best_acc


# ========================= MAIN =========================
def run_experiment(dataset_name, model, processor):
    run_tag = f"{dataset_name}/dinov2_{MODEL_SIZE}"
    print(f"\n{'='*80}\nSTARTING: {run_tag}\n{'='*80}")
    set_seed(SEED)

    ab_vals = [1, 2, 3, 4, 5]
    run_dir = OUTDIR / dataset_name / f"dinov2_{MODEL_SIZE}"
    run_dir.mkdir(parents=True, exist_ok=True)

    step_energy_gz = run_dir / "step_energy.csv.gz"
    metrics_csv    = run_dir / "epoch_metrics.csv"
    ensure_metrics_csv_header(ab_vals, metrics_csv)

    train_loader, test_loader, num_classes = make_loaders(dataset_name)
    print(f"[{dataset_name.upper()}] {num_classes} classes | "
          f"{len(train_loader.dataset)} train / {len(test_loader.dataset)} test | "
          f"batch={EVAL_BATCH_SIZE}")

    pwr = GpuPowerMeter(device_index=GPU_INDEX, step_energy_path=step_energy_gz)

    try:
        # ---- Feature extraction ----
        print(">>> Extracting train features...")
        pwr.reset_epoch()
        t0_feat = time.time()
        train_feats, train_labels = extract_features(
            model, processor, train_loader, pwr, phase="train", epoch_idx=0)
        print(">>> Extracting test features...")
        test_feats, test_labels = extract_features(
            model, processor, test_loader, pwr, phase="eval", epoch_idx=0)
        feat_time = time.time() - t0_feat
        feat_totals = pwr.epoch_totals()
        print(f"    train_feats: {train_feats.shape}  test_feats: {test_feats.shape}")
        print(f"    Feature extraction: {feat_time:.1f}s | Energy: {feat_totals['total_energy_j']:.0f}J")

        # ---- k-NN ----
        print(f"\n>>> k-NN classification (k={KNN_K})...")
        pwr.reset_epoch()
        t0_knn = time.time()
        knn_acc = knn_classify(train_feats, train_labels, test_feats, test_labels, k=KNN_K)
        knn_time = time.time() - t0_knn

        total_energy_knn = feat_totals['total_energy_j']
        total_time_knn = feat_time + knn_time
        print(f"    [k-NN] Acc={knn_acc:.2f}% | Time={total_time_knn:.1f}s | "
              f"Energy={total_energy_knn:.0f}J")

        sam_vals = compute_sam(knn_acc, total_energy_knn, ab_vals)
        row = [0, "knn", 0.0, f"{total_time_knn:.3f}", f"{total_time_knn:.3f}",
               0.0, f"{total_energy_knn:.3f}", f"{total_energy_knn:.3f}",
               f"{feat_totals['avg_power_w']:.3f}", f"{knn_acc:.2f}"]
        for a in ab_vals:
            v = sam_vals[f"SAM_a{a}_b{a}"]
            row.append(f"{v:.6f}" if not math.isnan(v) else "nan")
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

        # ---- Linear Probe ----
        print(f"\n>>> Linear probe ({LINEAR_EPOCHS} epochs, lr={LINEAR_LR})...")
        pwr.reset_epoch()
        t0_lp = time.time()
        lp_acc = linear_probe(train_feats, train_labels, test_feats, test_labels,
                              num_classes, pwr, epochs=LINEAR_EPOCHS, lr=LINEAR_LR)
        lp_time = time.time() - t0_lp
        lp_totals = pwr.epoch_totals()

        total_energy_lp = feat_totals['total_energy_j'] + lp_totals['total_energy_j']
        total_time_lp = feat_time + lp_time
        avg_power_lp = total_energy_lp / total_time_lp if total_time_lp > 0 else float("nan")
        print(f"    [Linear] Acc={lp_acc:.2f}% | Time={total_time_lp:.1f}s | "
              f"Energy={total_energy_lp:.0f}J")

        sam_vals = compute_sam(lp_acc, total_energy_lp, ab_vals)
        row = [LINEAR_EPOCHS, "linear", f"{lp_totals['train_time_s']:.3f}",
               f"{lp_totals['eval_time_s']:.3f}", f"{total_time_lp:.3f}",
               f"{feat_totals['total_energy_j'] + lp_totals['train_energy_j']:.3f}",
               f"{lp_totals['eval_energy_j']:.3f}", f"{total_energy_lp:.3f}",
               f"{avg_power_lp:.3f}", f"{lp_acc:.2f}"]
        for a in ab_vals:
            v = sam_vals[f"SAM_a{a}_b{a}"]
            row.append(f"{v:.6f}" if not math.isnan(v) else "nan")
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    finally:
        pwr.close()
        torch.cuda.empty_cache()

    print(f"\n[DONE] {run_tag} -> {metrics_csv}")
    return metrics_csv


if __name__ == "__main__":
    # Load model ONCE, reuse for all datasets
    model, processor = build_dinov2(MODEL_SIZE)

    results = {}
    for ds in DATASETS:
        try:
            csv_path = run_experiment(ds, model, processor)
            results[ds] = csv_path
        except Exception as e:
            print(f"\n[ERROR] {ds} failed: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

    import pandas as pd
    for ds, csv_path in results.items():
        print(f"\n{'='*60}")
        print(f"  {ds.upper()} - DINOv2-{MODEL_SIZE.upper()}")
        print(f"{'='*60}")
        df = pd.read_csv(csv_path)
        print(df.to_string(index=False))
        print(f"\nk-NN accuracy:     {df[df['method']=='knn']['test_acc_pct'].max():.2f}%")
        print(f"Linear probe acc:  {df[df['method']=='linear']['test_acc_pct'].max():.2f}%")
        print(f"Total energy:      {df['total_energy_j'].sum():.0f} J")
