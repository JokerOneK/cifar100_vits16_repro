"""
SigLIP Zero-Shot Evaluation on CIFAR-100 / DTD
Adapted from siglip-works-correctly.ipynb for local GPU (RTX 2060 Super 8GB).

Changes vs Kaggle notebook:
  - Paths: local instead of /kaggle/working/
  - Batch size: 32 (safe for 8 GB VRAM)
  - num_workers: 0 (Windows)
  - No pip install cell
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from PIL import Image
from transformers import SiglipModel, SiglipProcessor
import torchvision

print(f"torch={torch.__version__}  torchvision={torchvision.__version__}  "
      f"CUDA={torch.version.cuda}")

# ========================= CONFIG =========================
MODEL_SIZE      = "large"                # "base" | "large" | "so400m"
DATASETS        = ["cifar100", "dtd"]
EVAL_BATCH_SIZE = 32                     # 32 is safe for 8 GB; try 48 if OK
SEED            = 42
# ==========================================================

# Local paths (instead of /kaggle/working/)
SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR     = SCRIPT_DIR / "siglip_results"
DATA_ROOT  = SCRIPT_DIR / "data"
GPU_INDEX  = 0

SIGLIP_MODELS = {
    "base":   {"id": "google/siglip-base-patch16-224",   "params_m": 86},
    "large":  {"id": "google/siglip-large-patch16-384",  "params_m": 307},
    "so400m": {"id": "google/siglip-so400m-patch14-384", "params_m": 400},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram_gb:.1f} GB")

# ========================= CLASSES =========================
CIFAR100_CLASSES = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle",
    "bicycle","bottle","bowl","boy","bridge","bus","butterfly","camel",
    "can","castle","caterpillar","cattle","chair","chimpanzee","clock",
    "cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster",
    "house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion",
    "lizard","lobster","man","maple_tree","motorcycle","mountain","mouse",
    "mushroom","oak_tree","orange","orchid","otter","palm_tree","pear",
    "pickup_truck","pine_tree","plain","plate","poppy","porcupine",
    "possum","rabbit","raccoon","ray","road","rocket","rose","sea",
    "seal","shark","shrew","skunk","skyscraper","snail","snake",
    "spider","squirrel","streetcar","sunflower","sweet_pepper","table",
    "tank","telephone","television","tiger","tractor","train","trout",
    "tulip","turtle","wardrobe","whale","willow_tree","wolf","woman",
    "worm",
]

DTD_CLASSES = [
    "banded","blotchy","braided","bubbly","bumpy","chequered","cobwebbed",
    "cracked","crosshatched","crystalline","dotted","fibrous","flecked",
    "freckled","frilly","gauzy","grid","grooved","honeycombed","interlaced",
    "knitted","lacelike","lined","marbled","matted","meshed","paisley",
    "perforated","pitted","pleated","polka-dotted","porous","potholed",
    "scaly","smeared","spiralled","sprinkled","stained","stratified",
    "striped","studded","swirly","veined","waffled","woven","wrinkled",
    "zigzagged",
]

CIFAR100_TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.",
    "an image of a {}.",
    "a photograph of {}.",
    "a photo of the {}.",
]
DTD_TEMPLATES = [
    "a texture that is {}.",
    "a photo of a {} texture.",
    "a {} pattern.",
    "a surface that looks {}.",
    "{}.",
]


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


# ========================= DATASET =========================
class SiglipDataset(Dataset):
    def __init__(self, base_dataset, class_names):
        self.base = base_dataset
        self.class_names = class_names
        if hasattr(self.base, 'transform'): self.base.transform = None

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if isinstance(img, torch.Tensor): img = transforms.ToPILImage()(img)
        elif not isinstance(img, Image.Image): img = Image.fromarray(np.array(img))
        if img.mode != "RGB": img = img.convert("RGB")
        return img, label, self.class_names[label]


def siglip_collate(batch):
    images, labels, names = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(names)


def make_test_loader(dataset_name):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if dataset_name == "cifar100":
        ds = datasets.CIFAR100(root=str(DATA_ROOT), train=False,
                               transform=None, download=True)
        class_names = CIFAR100_CLASSES
        templates = CIFAR100_TEMPLATES
    elif dataset_name == "dtd":
        ds = datasets.DTD(root=str(DATA_ROOT), split="test",
                          transform=None, download=True)
        class_names = DTD_CLASSES
        templates = DTD_TEMPLATES
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    if hasattr(ds, "classes"):
        ds_classes = list(ds.classes)
        if ds_classes != class_names:
            print(f"[WARN] Using ds.classes instead of hardcoded list.")
            class_names = ds_classes

    wrapped = SiglipDataset(ds, class_names)
    loader  = DataLoader(wrapped, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                         num_workers=0,  # Windows
                         collate_fn=siglip_collate,
                         pin_memory=torch.cuda.is_available())
    return loader, class_names, templates


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
            h = ["epoch","train_time_s","eval_time_s","total_time_s",
                 "train_energy_j","eval_energy_j","total_energy_j","avg_power_w","test_acc_pct"]
            for a in ab_values: h.append(f"SAM_a{a}_b{a}")
            csv.writer(f).writerow(h)


# ========================= MODEL =========================
@torch.no_grad()
def build_class_embeddings(model, processor, class_names, templates, text_bs=64):
    all_texts = [t.format(c.replace("_"," ")) for c in class_names for t in templates]
    all_feats = []
    for i in range(0, len(all_texts), text_bs):
        batch_texts = all_texts[i:i+text_bs]
        inp = processor.tokenizer(
            batch_texts, return_tensors="pt",
            padding="max_length", truncation=True,
        ).to(device)
        f = model.get_text_features(**inp).float()
        all_feats.append(F.normalize(f, dim=-1).cpu())
    all_feats = torch.cat(all_feats)
    N, T = len(class_names), len(templates)
    all_feats = all_feats.view(N, T, -1).mean(dim=1)
    return F.normalize(all_feats, dim=-1).to(device)


def build_siglip(model_size):
    info = SIGLIP_MODELS[model_size]
    print(f"[SigLIP] Loading {info['id']} (~{info['params_m']}M params)...")
    processor = SiglipProcessor.from_pretrained(info["id"])
    model = SiglipModel.from_pretrained(
        info["id"], torch_dtype=torch.float16
    ).to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[SigLIP] Total params: {total_params:.1f}M")
    return model, processor


def _maybe_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def run_eval_discriminative(model, processor, test_loader, class_names,
                             text_embs, pwr, epoch_idx=-1):
    model.eval()
    correct = total = 0
    start_t = time.time()

    pbar = tqdm(enumerate(test_loader), total=len(test_loader),
                desc="Evaluating", leave=True)
    for step, (images, labels, _) in pbar:
        _maybe_sync()
        p_start = pwr.sample_power_w()
        t0 = time.time()

        inp = processor(images=images, return_tensors="pt").to(device)
        pixel_values = inp["pixel_values"].to(dtype=model.dtype)
        img_f = model.get_image_features(pixel_values=pixel_values).float()
        img_f = F.normalize(img_f, dim=-1)
        sims  = img_f @ text_embs.T
        preds = sims.argmax(dim=-1).cpu()

        correct += (preds == labels).sum().item()
        total   += len(labels)

        _maybe_sync()
        step_t = time.time() - t0
        pwr.log_step("eval", epoch_idx, step, step_t, p_start, pwr.sample_power_w())

        acc_so_far = 100.0 * correct / total if total > 0 else 0.0
        pbar.set_postfix(acc=f"{acc_so_far:.1f}%", n=total)

    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, time.time() - start_t


# ========================= MAIN =========================
def run_experiment(dataset_name):
    run_tag = f"{dataset_name}/siglip_{MODEL_SIZE}"
    print(f"\n{'='*80}\nSTARTING: {run_tag}\n{'='*80}")
    set_seed(SEED)

    ab_vals = [1, 2, 3, 4, 5]
    run_dir = OUTDIR / dataset_name / f"siglip_{MODEL_SIZE}"
    run_dir.mkdir(parents=True, exist_ok=True)

    step_energy_gz = run_dir / "step_energy.csv.gz"
    metrics_csv    = run_dir / "epoch_metrics.csv"
    ensure_metrics_csv_header(ab_vals, metrics_csv)

    test_loader, class_names, templates = make_test_loader(dataset_name)
    print(f"[{dataset_name.upper()}] {len(class_names)} classes | "
          f"{len(test_loader.dataset)} test images | batch={EVAL_BATCH_SIZE} | "
          f"{len(test_loader)} batches")

    model, processor = build_siglip(MODEL_SIZE)
    pwr = GpuPowerMeter(device_index=GPU_INDEX, step_energy_path=step_energy_gz)

    try:
        print(">>> Building class text embeddings...")
        text_embs = build_class_embeddings(model, processor, class_names, templates)
        print(f"    text_embs: {text_embs.shape}")

        print(">>> Zero-shot evaluation...")
        pwr.reset_epoch()
        acc, eval_time = run_eval_discriminative(
            model, processor, test_loader, class_names, text_embs, pwr, epoch_idx=0)

        totals = pwr.epoch_totals()
        print(f"\n[ZERO-SHOT] Acc={acc:.2f}% | Time={eval_time:.1f}s | "
              f"AvgW={totals['avg_power_w']:.1f} | Energy={totals['total_energy_j']:.0f}J")

        sam_vals = compute_sam(acc, totals["total_energy_j"], ab_vals)
        row = [0, 0.0, f"{eval_time:.3f}", f"{eval_time:.3f}",
               0.0, f"{totals['eval_energy_j']:.3f}", f"{totals['total_energy_j']:.3f}",
               f"{totals['avg_power_w']:.3f}", f"{acc:.2f}"]
        for a in ab_vals:
            v = sam_vals[f"SAM_a{a}_b{a}"]
            row.append(f"{v:.6f}" if not math.isnan(v) else "nan")
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    finally:
        pwr.close()
        torch.cuda.empty_cache()

    print(f"[DONE] {run_tag} -> {metrics_csv}")
    return metrics_csv


if __name__ == "__main__":
    results = {}
    for ds in DATASETS:
        try:
            csv_path = run_experiment(ds)
            results[ds] = csv_path
        except Exception as e:
            print(f"\n[ERROR] {ds} failed: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

    # Print summary
    import pandas as pd
    for ds, csv_path in results.items():
        print(f"\n{'='*60}")
        print(f"  {ds.upper()} - SigLIP-{MODEL_SIZE.upper()}")
        print(f"{'='*60}")
        df = pd.read_csv(csv_path)
        print(df.to_string(index=False))
        print(f"\nBest accuracy: {df['test_acc_pct'].max():.2f}%")
        print(f"Total energy:  {df['total_energy_j'].sum():.0f} J")
        print(f"Total time:    {df['total_time_s'].sum()/60:.1f} min")
