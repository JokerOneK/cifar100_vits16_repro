"""
SmolVLM generative classifier baseline for CIFAR-100 / DTD.

Instead of a discriminative head (softmax over classes), SmolVLM receives
an image + text prompt listing possible classes and generates the class name.
The model is PRETRAINED (instruction-tuned) and works zero-shot out of the box.
Optionally fine-tuned via SFT (supervised fine-tuning) on classification data.

Usage:
    # Zero-shot only (no training)
    python train_smolvlm_textures.py --datasets cifar100 --epochs 0

    # SFT fine-tuning, 5 epochs
    python train_smolvlm_textures.py --datasets dtd --epochs 5

    # Both datasets
    python train_smolvlm_textures.py --datasets cifar100 dtd --epochs 5

    # Larger model
    python train_smolvlm_textures.py --model-size 500m --datasets cifar100
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import math
import time
import gzip
import csv
import argparse
import random
import traceback
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, ConcatDataset
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image

# --- HuggingFace transformers ---
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERROR: 'transformers' library not found. Install: pip install transformers accelerate")

# ===============================
# Defaults
# ===============================
EPOCHS = 0
STEPS_PER_EPOCH = 200
BATCH_SIZE = 4
NUM_WORKERS = 0        # Windows-safe
SEED = 42
DEFAULT_OUTDIR = Path('./smolvlm_results')
DEFAULT_DATA_ROOT = Path('./data')
GPU_INDEX = 0

SMOLVLM_MODELS = {
    "256m": {"id": "HuggingFaceTB/SmolVLM-256M-Instruct", "params_m": 256},
    "500m": {"id": "HuggingFaceTB/SmolVLM-500M-Instruct", "params_m": 500},
    "2b":   {"id": "HuggingFaceTB/SmolVLM-Instruct",      "params_m": 2200},
}

# ===============================
# Class name lists
# ===============================
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]

DTD_CLASSES = [
    "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed",
    "cracked", "crosshatched", "crystalline", "dotted", "fibrous", "flecked",
    "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed", "interlaced",
    "knitted", "lacelike", "lined", "marbled", "matted", "meshed", "paisley",
    "perforated", "pitted", "pleated", "polka-dotted", "porous", "potholed",
    "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified",
    "striped", "studded", "swirly", "veined", "waffled", "woven", "wrinkled",
    "zigzagged",
]

DTD_DEFINITIONS = {
    "banded":       "parallel stripes of alternating colors",
    "blotchy":      "irregular patches of color on surface",
    "braided":      "interlaced strands woven together",
    "bubbly":       "rounded bubble-like circular forms",
    "bumpy":        "raised rounded protrusions on surface",
    "chequered":    "alternating squares like a checkerboard",
    "cobwebbed":    "thin threads forming a web pattern",
    "cracked":      "irregular fracture lines on surface",
    "crosshatched": "overlapping parallel lines at angles",
    "crystalline":  "faceted geometric crystal shapes",
    "dotted":       "small round spots on surface",
    "fibrous":      "thin parallel thread-like fibers",
    "flecked":      "small specks or flakes scattered",
    "freckled":     "small spots irregularly scattered",
    "frilly":       "ruffled wavy edges or folds",
    "gauzy":        "thin translucent loosely woven material",
    "grid":         "regular rectangular grid lines",
    "grooved":      "parallel channels carved in surface",
    "honeycombed":  "hexagonal cells like a beehive",
    "interlaced":   "overlapping woven crossing elements",
    "knitted":      "looped yarn with visible stitches",
    "lacelike":     "delicate openwork with intricate holes",
    "lined":        "parallel straight lines",
    "marbled":      "swirling veins like marble stone",
    "matted":       "tangled compressed flat fibers",
    "meshed":       "open net-like grid with holes",
    "paisley":      "curved teardrop decorative shapes",
    "perforated":   "surface with many small holes",
    "pitted":       "small indentations or pores",
    "pleated":      "parallel fabric folds",
    "polka-dotted": "regular round spots on background",
    "porous":       "sponge-like with many irregular pores",
    "potholed":     "large irregular holes or depressions",
    "scaly":        "overlapping scales like fish skin",
    "smeared":      "blurred streaks spread across surface",
    "spiralled":    "curved lines winding from a center",
    "sprinkled":    "small particles randomly scattered",
    "stained":      "discolored patches from absorbed liquid",
    "stratified":   "visible horizontal layers stacked",
    "striped":      "alternating parallel color bands",
    "studded":      "raised bumps regularly spaced",
    "swirly":       "flowing curved circular lines",
    "veined":       "thin branching lines like veins",
    "waffled":      "raised squares with recessed grid",
    "woven":        "interlaced threads forming fabric",
    "wrinkled":     "irregular folds and creases",
    "zigzagged":    "sharp V-shaped angular lines",
}


# ========================= CIFAR-100 HIERARCHY =========================
CIFAR100_SUPERCLASSES = [
    "aquatic_mammals", "fish", "flowers", "food_containers",
    "fruit_and_vegetables", "household_electrical_devices", "household_furniture",
    "insects", "large_carnivores", "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
    "medium_mammals", "non-insect_invertebrates", "people", "reptiles",
    "small_mammals", "trees", "vehicles_1", "vehicles_2",
]

CIFAR100_HIERARCHY = {
    "aquatic_mammals":                ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish":                           ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers":                        ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers":                ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables":           ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices":   ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture":            ["bed", "chair", "couch", "table", "wardrobe"],
    "insects":                        ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores":               ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man-made_outdoor_things":  ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes":   ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_mammals":                 ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect_invertebrates":       ["crab", "lobster", "snail", "spider", "worm"],
    "people":                         ["baby", "boy", "girl", "man", "woman"],
    "reptiles":                       ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals":                  ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees":                          ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1":                     ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2":                     ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}

FINE_TO_SUPERCLASS = {
    fine: sc
    for sc, fines in CIFAR100_HIERARCHY.items()
    for fine in fines
}

SUPERCLASS_PROMPT = (
    "What broad category does this image belong to? Choose exactly one from: "
    "aquatic mammals, fish, flowers, food containers, fruit and vegetables, "
    "household electrical devices, household furniture, insects, large carnivores, "
    "large man-made outdoor things, large natural outdoor scenes, "
    "large omnivores and herbivores, medium mammals, non-insect invertebrates, "
    "people, reptiles, small mammals, trees, vehicles 1, vehicles 2. "
    "Reply with ONLY the category name, nothing else."
)

def build_fine_prompt(fine_classes):
    class_list = ", ".join(c.replace("_", " ") for c in fine_classes)
    return (f"What specific object is shown? Choose exactly one from: "
            f"{class_list}. Reply with ONLY the object name, nothing else.")

def match_superclass(gen_text):
    gen = gen_text.strip().lower().replace("_", " ").replace("-", " ")
    sc_display = {sc: sc.replace("_", " ").replace("-", " ") for sc in CIFAR100_SUPERCLASSES}
    for sc, disp in sc_display.items():
        if gen == disp or disp in gen or gen in disp:
            return sc
    best_sc, best_score = CIFAR100_SUPERCLASSES[0], 0.0
    for sc, disp in sc_display.items():
        score = SequenceMatcher(None, gen, disp).ratio()
        if score > best_score:
            best_score, best_sc = score, sc
    return best_sc

def match_fine_class(gen_text, fine_classes, all_class_names):
    gen = gen_text.strip().lower().replace("_", " ")
    for cls in fine_classes:
        cls_c = cls.lower().replace("_", " ")
        if gen == cls_c or cls_c in gen or gen in cls_c:
            return all_class_names.index(cls)
    best_cls, best_score = fine_classes[0], 0.0
    for cls in fine_classes:
        cls_c = cls.lower().replace("_", " ")
        score = SequenceMatcher(None, gen, cls_c).ratio()
        if score > best_score:
            best_score, best_cls = score, cls
    return all_class_names.index(best_cls)


def iso_now():
    return datetime.now().isoformat(timespec="seconds")


def set_seed(seed=SEED):
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
        raise RuntimeError("CUDA GPU not found.")
    return torch.device("cuda")


# ===============================
# Prompt construction
# ===============================
def build_classification_prompt(class_names: List[str], dataset_name: str,
                               use_definitions: bool = False) -> str:
    if dataset_name == "cifar100":
        class_list = ", ".join(class_names)
        return (
            f"Classify this image into exactly one of the following {len(class_names)} categories: "
            f"{class_list}. "
            f"Reply with ONLY the category name, nothing else."
        )
    else:  # dtd
        if use_definitions:
            lines = ["What texture pattern is shown? Choose exactly one from the list below."]
            for cls in class_names:
                defn = DTD_DEFINITIONS.get(cls, "")
                lines.append(f"  {cls}: {defn}")
            lines.append("Reply with ONLY the texture name, nothing else.")
            return "\n".join(lines)
        else:
            class_list = ", ".join(class_names)
            return (
                f"What is the texture pattern in this image? Choose exactly one from: "
                f"{class_list}. "
                f"Reply with ONLY the texture name, nothing else."
            )


def match_prediction_to_class(pred_text: str, class_names: List[str]) -> Tuple[int, str]:
    """Match generated text to a known class name (exact -> prefix -> contains -> fuzzy)."""
    pred_clean = pred_text.strip().lower().replace("_", " ").replace("-", " ")

    # 1) Exact match
    for i, cls in enumerate(class_names):
        cls_clean = cls.lower().replace("_", " ").replace("-", " ")
        if pred_clean == cls_clean:
            return i, cls

    # 2) Prediction starts with class name
    for i, cls in enumerate(class_names):
        cls_clean = cls.lower().replace("_", " ").replace("-", " ")
        if pred_clean.startswith(cls_clean):
            return i, cls

    # 3) Class name contained in prediction
    for i, cls in enumerate(class_names):
        cls_clean = cls.lower().replace("_", " ").replace("-", " ")
        if cls_clean in pred_clean:
            return i, cls

    # 4) Fuzzy matching
    best_ratio = 0.0
    best_idx = 0
    for i, cls in enumerate(class_names):
        cls_clean = cls.lower().replace("_", " ").replace("-", " ")
        ratio = SequenceMatcher(None, pred_clean, cls_clean).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i
    return best_idx, class_names[best_idx]


# ===============================
# VLM Dataset wrapper
# ===============================
class VLMClassificationDataset(Dataset):
    """Wraps a torchvision dataset. Returns (PIL image, label_idx, class_name)."""
    def __init__(self, base_dataset, class_names: List[str]):
        self.base = base_dataset
        self.class_names = class_names
        # Disable transforms — VLM processor handles image preprocessing
        if hasattr(self.base, 'transform'):
            self.base.transform = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img, label, self.class_names[label]


def vlm_collate_fn(batch):
    images, labels, names = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(names)


def make_vlm_dataloaders_cifar100(steps_per_epoch, batch_size, num_workers,
                                   data_root, download, eval_batch_size=4):
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    train_ds = datasets.CIFAR100(root=str(root), train=True, transform=None, download=download)
    test_ds = datasets.CIFAR100(root=str(root), train=False, transform=None, download=download)
    class_names = CIFAR100_CLASSES

    train_vlm = VLMClassificationDataset(train_ds, class_names)
    test_vlm = VLMClassificationDataset(test_ds, class_names)

    num_samples = steps_per_epoch * batch_size
    train_sampler = RandomSampler(train_vlm, replacement=True, num_samples=num_samples)

    train_loader = DataLoader(train_vlm, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=False, drop_last=True,
                              collate_fn=vlm_collate_fn)
    test_loader = DataLoader(test_vlm, batch_size=eval_batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False,
                             collate_fn=vlm_collate_fn)
    return train_loader, test_loader, len(class_names), class_names


def make_vlm_dataloaders_dtd(steps_per_epoch, batch_size, num_workers,
                              data_root, download, eval_batch_size=4):
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    dtd_train = datasets.DTD(root=str(root), split="train", transform=None, download=download)
    dtd_val = datasets.DTD(root=str(root), split="val", transform=None, download=download)
    dtd_test = datasets.DTD(root=str(root), split="test", transform=None, download=download)
    class_names = DTD_CLASSES

    train_vlm = VLMClassificationDataset(ConcatDataset([dtd_train, dtd_val]), class_names)
    test_vlm = VLMClassificationDataset(dtd_test, class_names)

    num_samples = steps_per_epoch * batch_size
    train_sampler = RandomSampler(train_vlm, replacement=True, num_samples=num_samples)

    train_loader = DataLoader(train_vlm, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, pin_memory=False, drop_last=True,
                              collate_fn=vlm_collate_fn)
    test_loader = DataLoader(test_vlm, batch_size=eval_batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False,
                             collate_fn=vlm_collate_fn)
    return train_loader, test_loader, len(class_names), class_names


# ===============================
# Power/Energy logger
# ===============================
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

    def _accumulate(self, phase, step_time_s, p_start, p_end):
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

    def log_step(self, phase, epoch, step, step_time_s, p_start, p_end):
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


# ===============================
# Metrics helpers
# ===============================
class SmoothedValue:
    def __init__(self, momentum=0.98):
        self.m = None
        self.beta = momentum
    def update(self, x):
        self.m = x if self.m is None else self.beta * self.m + (1 - self.beta) * x
    @property
    def value(self):
        return float(self.m) if self.m is not None else float("nan")


def safe_log10(x):
    if x is None or math.isnan(x) or x <= 0:
        return float("nan")
    return math.log10(x)


def compute_sam(acc_pct, energy_j, ab_values):
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


def ensure_metrics_csv_header(ab_values, metrics_path):
    first = (not metrics_path.exists()) or (metrics_path.stat().st_size == 0)
    if first:
        with open(metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            header = ["epoch", "train_time_s", "eval_time_s", "total_time_s",
                      "train_energy_j", "eval_energy_j", "total_energy_j",
                      "avg_power_w", "test_acc_pct"]
            for a in ab_values:
                header.append(f"SAM_a{a}_b{a}")
            w.writerow(header)


# ===============================
# Model loading
# ===============================
def build_smolvlm(model_size: str, device):
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers required. pip install transformers accelerate")

    info = SMOLVLM_MODELS[model_size]
    model_id = info["id"]
    print(f"[SmolVLM] Loading {model_id} (~{info['params_m']}M params) ...")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # Fix padding warning for decoder-only architecture
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)


    total_params = sum(p.numel() for p in model.parameters())
    print(f"[SmolVLM] Total params: {total_params / 1e6:.1f}M")

    return model, processor


# ===============================
# Training step (SFT)
# ===============================
def train_step_sft(model, processor, images, class_names_batch, prompt_text, device):
    """SFT step: image + prompt -> model learns to generate the correct class name."""
    texts = []
    for class_name in class_names_batch:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": class_name},
                ],
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False)
        texts.append(text)

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss


# ===============================
# Evaluation (generative)
# ===============================
@torch.no_grad()
def run_eval_generative(model, processor, test_loader, class_names, prompt_text,
                         device, pwr, epoch_idx=-1, max_new_tokens=5):
    model.eval()
    correct = 0
    total = 0
    start_t = time.time()

    MAX_IMG_SIZE = 256  # cap image size before passing to processor to avoid tiling OOM

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating", leave=True)
    for step, (images, labels, _) in pbar:
        torch.cuda.synchronize()
        p_start = pwr.sample_power_w()
        t0 = time.time()

        # Resize large images to prevent SmolVLM from creating too many tiles
        images = [
            img.resize((MAX_IMG_SIZE, MAX_IMG_SIZE), Image.BILINEAR)
            if max(img.size) > MAX_IMG_SIZE else img
            for img in images
        ]

        texts = []
        for _ in images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
            texts.append(text)

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
        )

        # Decode only the generated part
        input_len = inputs["input_ids"].shape[1]
        generated_texts = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )

        for pred_text, label in zip(generated_texts, labels):
            pred_idx, _ = match_prediction_to_class(pred_text, class_names)
            if pred_idx == label.item():
                correct += 1
            total += 1

        torch.cuda.synchronize()
        step_t = time.time() - t0
        p_end = pwr.sample_power_w()
        pwr.log_step("eval", epoch_idx, step, step_t, p_start, p_end)
        torch.cuda.empty_cache()

        acc_so_far = 100.0 * correct / total if total > 0 else 0.0
        pbar.set_postfix(acc=f"{acc_so_far:.1f}%", n=total)

    total_time = time.time() - start_t
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, total_time


@torch.no_grad()
def run_eval_hierarchical_smolvlm(model, processor, test_ds_base, class_names,
                                   device, pwr, epoch_idx=-1, max_new_tokens=5):
    """Two-step CIFAR-100 eval for SmolVLM: superclass → fine class."""
    model.eval()
    correct = total = sc_correct = 0
    MAX_IMG_SIZE = 256
    start_t = time.time()
    n_samples = len(test_ds_base)
    pbar = tqdm(range(n_samples), desc="Evaluating (hierarchical)", leave=True)

    def _infer(img_pil, prompt_text):
        img_pil = (img_pil.resize((MAX_IMG_SIZE, MAX_IMG_SIZE), Image.BILINEAR)
                   if max(img_pil.size) > MAX_IMG_SIZE else img_pil)
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt_text}]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[img_pil],
                           return_tensors="pt", padding=True).to(device)
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, use_cache=False)
        out = processor.batch_decode(gen_ids[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        torch.cuda.empty_cache()
        return out[0].strip()

    for step in pbar:
        torch.cuda.synchronize()
        p_start = pwr.sample_power_w()
        t0 = time.time()

        img, label = test_ds_base[step]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        if img.mode != "RGB":
            img = img.convert("RGB")

        true_sc = FINE_TO_SUPERCLASS.get(class_names[label])

        # Step 1: superclass
        sc_gen  = _infer(img, SUPERCLASS_PROMPT)
        sc_pred = match_superclass(sc_gen)
        if true_sc and sc_pred == true_sc:
            sc_correct += 1

        # Step 2: fine class
        fine_classes = CIFAR100_HIERARCHY[sc_pred]
        fine_gen  = _infer(img, build_fine_prompt(fine_classes))
        pred_idx  = match_fine_class(fine_gen, fine_classes, class_names)

        if pred_idx == label:
            correct += 1
        total += 1

        torch.cuda.synchronize()
        step_t = time.time() - t0
        pwr.log_step("eval", epoch_idx, step, step_t, p_start, pwr.sample_power_w())

        acc_so_far = 100.0 * correct    / total if total > 0 else 0.0
        sc_acc     = 100.0 * sc_correct / total if total > 0 else 0.0
        pbar.set_postfix(acc=f"{acc_so_far:.1f}%", sc=f"{sc_acc:.1f}%",
                         s_img=f"{step_t:.1f}s")

    acc    = 100.0 * correct    / total if total > 0 else 0.0
    sc_acc = 100.0 * sc_correct / total if total > 0 else 0.0
    elapsed = time.time() - start_t
    print(f"\nSuperclass acc: {sc_acc:.2f}%  |  Final acc: {acc:.2f}%")
    return acc, elapsed


# ===============================
# Single run
# ===============================
def train_single_run(args, dataset: str):
    run_tag = f"{dataset}/smolvlm_{args.model_size}"
    print(f"\n{'=' * 80}")
    print(f"STARTING: {run_tag}")
    print(f"{'=' * 80}")
    set_seed(SEED)
    device = ensure_cuda()

    ab_vals = sorted(set(int(s) for s in args.sam_ab.split(",") if s.strip()))

    run_dir = Path(args.outdir) / dataset / f"smolvlm_{args.model_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN DIR] {run_dir}")

    step_energy_gz = run_dir / "step_energy.csv.gz"
    metrics_csv_path = run_dir / "epoch_metrics.csv"
    ensure_metrics_csv_header(ab_vals, metrics_csv_path)

    # --- Dataset ---
    if dataset == "cifar100":
        train_loader, test_loader, num_classes, class_names = make_vlm_dataloaders_cifar100(
            args.steps_per_epoch, args.batch_size, args.num_workers,
            data_root=args.data_root, download=args.download,
            eval_batch_size=args.eval_batch_size,
        )
    else:
        train_loader, test_loader, num_classes, class_names = make_vlm_dataloaders_dtd(
            args.steps_per_epoch, args.batch_size, args.num_workers,
            data_root=args.data_root, download=args.download,
            eval_batch_size=args.eval_batch_size,
        )
    print(f"[{dataset.upper()}] num_classes={num_classes}")

    prompt_text = build_classification_prompt(class_names, dataset, args.use_definitions)

    # --- Model ---
    model, processor = build_smolvlm(args.model_size, device)

    pwr = GpuPowerMeter(device_index=args.gpu_index, step_energy_path=step_energy_gz)

    try:
        # --- Zero-shot baseline ---
        use_hier = args.use_hierarchical and dataset == "cifar100"
        mode_tag = "HIERARCHICAL" if use_hier else "ZERO-SHOT"
        print(f">>> {mode_tag} evaluation...")
        pwr.reset_epoch()
        if use_hier:
            # Hierarchical needs per-image access — use base dataset directly
            from torchvision import datasets as _tv_ds
            from pathlib import Path as _Path
            _base_ds = _tv_ds.CIFAR100(root=str(_Path(args.data_root)), train=False,
                                       transform=None, download=args.download)
            base_acc, base_time = run_eval_hierarchical_smolvlm(
                model, processor, _base_ds, class_names, device, pwr, epoch_idx=0)
        else:
            base_acc, base_time = run_eval_generative(
                model, processor, test_loader, class_names, prompt_text,
                device, pwr, epoch_idx=0,
            )
        print(f"[{mode_tag}] Acc={base_acc:.2f}% Time={base_time:.2f}s")

        with open(metrics_csv_path, "a", newline="") as f:
            row = [0, 0.0, f"{base_time:.3f}", f"{base_time:.3f}",
                   0.0, 0.0, 0.0, 0.0, f"{base_acc:.2f}"]
            for _ in ab_vals:
                row.append("nan")
            csv.writer(f).writerow(row)

        # --- SFT training ---
        if args.epochs > 0:
            for p in model.parameters():
                p.requires_grad = True

            trainable_params = list(model.parameters())
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                           weight_decay=args.weight_decay)

            total_params_m = sum(p.numel() for p in trainable_params) / 1e6
            print(f"[SFT] LR={args.lr:.1e} | Trainable: {total_params_m:.1f}M params")

            for epoch in range(1, args.epochs + 1):
                pwr.reset_epoch()
                model.train()
                torch.cuda.reset_peak_memory_stats(device)
                loss_smooth = SmoothedValue(0.98)
                start_epoch_t = time.time()

                iterator = enumerate(train_loader, start=1)
                if not args.no_progress:
                    iterator = tqdm(iterator, total=args.steps_per_epoch, ncols=120,
                                    leave=False, desc=f"Epoch {epoch}")

                for step, (images, labels, names_batch) in iterator:
                    torch.cuda.synchronize()
                    p_start = pwr.sample_power_w()
                    t0 = time.time()

                    optimizer.zero_grad(set_to_none=True)
                    loss = train_step_sft(model, processor, images, names_batch,
                                          prompt_text, device)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()

                    torch.cuda.synchronize()
                    step_t = time.time() - t0
                    p_end = pwr.sample_power_w()
                    pwr.log_step("train", epoch, step, step_t, p_start, p_end)
                    loss_smooth.update(loss.item())

                    if step % args.log_interval == 0:
                        alloc = bytes_to_mib(torch.cuda.memory_allocated(device))
                        peak = bytes_to_mib(torch.cuda.max_memory_allocated(device))
                        print(f"[E{epoch:02d} S{step:05d}] loss={loss.item():.4f} "
                              f"sm={loss_smooth.value:.4f} "
                              f"alloc={alloc:.0f}MiB peak={peak:.0f}MiB")

                    if step >= args.steps_per_epoch:
                        break

                dt_train = time.time() - start_epoch_t

                acc, dt_eval = run_eval_generative(
                    model, processor, test_loader, class_names, prompt_text,
                    device, pwr, epoch_idx=epoch,
                )

                totals = pwr.epoch_totals()
                sam_vals = compute_sam(acc, totals["total_energy_j"], ab_vals)
                peak_epoch = bytes_to_mib(torch.cuda.max_memory_allocated(device))

                print(f"[Epoch {epoch}/{args.epochs}] Acc={acc:.2f}% "
                      f"TrainT={dt_train / 60:.1f}m EvalT={dt_eval:.1f}s "
                      f"AvgW={totals['avg_power_w']:.1f} "
                      f"Energy={totals['total_energy_j']:.0f}J "
                      f"PeakMem={peak_epoch:.0f}MiB")

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
        pwr.close()
        del model, processor
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print(f"\n[DONE] {run_tag}")


# ===============================
# CLI
# ===============================
def parse_args():
    ap = argparse.ArgumentParser(
        description="SmolVLM generative classifier baseline for CIFAR-100 / DTD"
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="SFT epochs (0 = zero-shot only)")
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--gpu-index", type=int, default=GPU_INDEX)
    ap.add_argument("--sam-ab", type=str, default="1,2,3,4,5")

    ap.add_argument("--model-size", type=str, default="256m",
                    choices=["256m", "500m", "2b"],
                    help="SmolVLM variant (default: 256m, ~0.5GB VRAM for weights)")

    ap.add_argument("--datasets", type=str, nargs="+", default=["dtd"],
                    choices=["cifar100", "dtd"])

    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--use-definitions", action="store_true",
                    default=os.environ.get("VLM_USE_DEFINITIONS", "0") == "1",
                    help="Use class-definition prompts for DTD")
    ap.add_argument("--use-hierarchical", action="store_true",
                    default=os.environ.get("VLM_USE_HIERARCHICAL", "0") == "1",
                    help="Use two-step hierarchical prompting for CIFAR-100")

    return ap.parse_args()


def main():
    args = parse_args()

    datasets_list = args.datasets
    total_runs = len(datasets_list)

    print(f"\n{'#' * 80}")
    print(f"SmolVLM BASELINE: {total_runs} run(s)")
    print(f"  Model:     SmolVLM-{args.model_size.upper()} ({SMOLVLM_MODELS[args.model_size]['params_m']}M params)")
    print(f"  Datasets:  {datasets_list}")
    print(f"  Epochs:    {args.epochs} ({'zero-shot only' if args.epochs == 0 else 'SFT'})")
    print(f"  Batch:     {args.batch_size} (train) / {args.eval_batch_size} (eval)")
    print(f"  Outdir:    {args.outdir}")
    print(f"{'#' * 80}\n")

    completed = 0
    failed = 0
    failed_runs = []

    for ds in datasets_list:
        run_tag = f"{ds}/smolvlm_{args.model_size}"
        run_idx = completed + failed + 1
        print(f"\n>>> [{run_idx}/{total_runs}] {run_tag}")
        try:
            train_single_run(args, dataset=ds)
            completed += 1
        except Exception as e:
            failed += 1
            failed_runs.append(run_tag)
            print(f"\n[ERROR] Run '{run_tag}' failed: {e}")
            traceback.print_exc()
            print(f"[SKIP] Continuing...\n")
            torch.cuda.empty_cache()
            continue

    print(f"\n{'#' * 80}")
    print(f"COMPLETE: {completed}/{total_runs} succeeded")
    if failed_runs:
        print(f"  Failed: {failed_runs}")
    print(f"{'#' * 80}")


if __name__ == "__main__":
    main()
