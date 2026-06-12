"""
MobileVLM V2-1.7B Generative Zero-Shot Classification on CIFAR-100 / DTD
Adapted from mobilevlm.ipynb for local GPU (RTX 2060 Super 8GB).

Changes vs Kaggle notebook:
  - Paths: local instead of /kaggle/working/
  - MobileVLM repo cloned to src/MobileVLM/ (or set MOBILEVLM_REPO env var)
  - num_workers: 0 (Windows)
  - No pip install cell

Setup (one-time):
  cd src
  git clone --depth 1 https://github.com/Meituan-AutoML/MobileVLM.git
  pip install transformers accelerate pynvml sentencepiece protobuf pandas

Run:
  python src/train_mobilevlm_textures.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:128,garbage_collection_threshold:0.8"
)

import sys, math, time, gzip, csv, random, traceback, re
from pathlib import Path
from datetime import datetime

# ---- Add MobileVLM repo to path ----
SCRIPT_DIR    = Path(__file__).resolve().parent
MOBILEVLM_DIR = Path(os.environ.get("MOBILEVLM_REPO", str(SCRIPT_DIR / "MobileVLM")))
if not MOBILEVLM_DIR.is_dir():
    raise RuntimeError(
        f"MobileVLM repo not found at {MOBILEVLM_DIR}\n"
        f"Clone it first:\n"
        f"  cd {SCRIPT_DIR}\n"
        f"  git clone --depth 1 https://github.com/Meituan-AutoML/MobileVLM.git\n"
        f"Or set MOBILEVLM_REPO env var to the repo path."
    )
if str(MOBILEVLM_DIR) not in sys.path:
    sys.path.insert(0, str(MOBILEVLM_DIR))

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from PIL import Image
import torchvision

print(f"torch={torch.__version__}  torchvision={torchvision.__version__}  "
      f"CUDA={torch.version.cuda}")

# MobileVLM imports (from cloned repo)
from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import (
    disable_torch_init, process_images,
    tokenizer_image_token, KeywordsStoppingCriteria,
)
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX

# ========================= CONFIG =========================
MODEL_PATH      = "mtgv/MobileVLM_V2-1.7B"
_ds_env         = os.environ.get("VLM_DATASETS", "cifar100,dtd")
DATASETS        = [d.strip() for d in _ds_env.split(",") if d.strip()]
USE_DEFINITIONS  = os.environ.get("VLM_USE_DEFINITIONS",  "0") == "1"
USE_HIERARCHICAL = os.environ.get("VLM_USE_HIERARCHICAL", "0") == "1"
SEED            = 42
CONV_MODE       = "v1"
MAX_NEW_TOKENS  = 16
MAX_SAMPLES     = None        # set e.g. 500 for quick test, None = full dataset
# --- Fine-tuning (mirrors train_paligemma_textures.py) ---
FT_MODE         = os.environ.get("VLM_FT_MODE", "zero_shot")  # "zero_shot" | "qlora"
FT_EPOCHS       = int(os.environ.get("VLM_FT_EPOCHS", "1"))
_default_ft_lr  = "2e-4" if FT_MODE == "qlora" else "2e-5"
FT_LR           = float(os.environ.get("VLM_FT_LR", _default_ft_lr))
FT_STEPS_PER_EPOCH = int(os.environ.get("VLM_FT_STEPS", "500"))
FT_SKIP_BASELINE = os.environ.get("VLM_FT_SKIP_BASELINE", "0") == "1"
FT_GRAD_ACCUM   = int(os.environ.get("VLM_FT_GRAD_ACCUM", "8"))
FT_LORA_R       = int(os.environ.get("VLM_FT_LORA_R", "16"))
FT_LORA_ALPHA   = int(os.environ.get("VLM_FT_LORA_ALPHA", "32"))
FT_LORA_DROPOUT = float(os.environ.get("VLM_FT_LORA_DROPOUT", "0.05"))
# light-touch: train with the eval prompt + early-stop on a held-out val subset
FT_PROMPT_MODE  = os.environ.get("VLM_FT_PROMPT", "eval")     # "eval" | "short"
FT_VAL_EVERY    = int(os.environ.get("VLM_FT_VAL_EVERY", "25"))
FT_VAL_SAMPLES  = int(os.environ.get("VLM_FT_VAL_SAMPLES", "120"))
FT_PATIENCE     = int(os.environ.get("VLM_FT_PATIENCE", "3"))
# ==========================================================

_outdir_env = os.environ.get("VLM_OUTDIR")
OUTDIR    = Path(_outdir_env) if _outdir_env else SCRIPT_DIR / "mobilevlm_results"
DATA_ROOT = SCRIPT_DIR / "data"
GPU_INDEX = 0

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

CIFAR100_PROMPT = (
    "Look at this image and classify it into exactly one of these categories: {}. "
    "Answer with ONLY the category name, nothing else."
)
DTD_PROMPT = (
    "What texture or pattern does this image show? Choose exactly one from: {}. "
    "Answer with ONLY the texture name, nothing else."
)

# Short prompts for FT_PROMPT_MODE="short" (no class list).
CIFAR100_FT_PROMPT = "What is in this image?"
DTD_FT_PROMPT      = "What texture is shown?"

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

def build_dtd_definitions_prompt(class_names):
    lines = ["What texture pattern is shown? Choose exactly one from the list below."]
    for cls in class_names:
        defn = DTD_DEFINITIONS.get(cls, "")
        lines.append(f"  {cls}: {defn}")
    lines.append("Reply with ONLY the texture name, nothing else.")
    return "\n".join(lines)


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
    from difflib import SequenceMatcher
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
    from difflib import SequenceMatcher
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
def load_test_dataset(dataset_name):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if dataset_name == "cifar100":
        ds = datasets.CIFAR100(root=str(DATA_ROOT), train=False,
                               transform=None, download=True)
        class_names = CIFAR100_CLASSES
        prompt_tpl = CIFAR100_PROMPT
    elif dataset_name == "dtd":
        ds = datasets.DTD(root=str(DATA_ROOT), split="test",
                          transform=None, download=True)
        class_names = DTD_CLASSES
        prompt_tpl = build_dtd_definitions_prompt(DTD_CLASSES) if USE_DEFINITIONS else DTD_PROMPT
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    if hasattr(ds, "classes"):
        ds_classes = list(ds.classes)
        if ds_classes != class_names:
            print(f"[WARN] Using ds.classes instead of hardcoded list.")
            class_names = ds_classes

    return ds, class_names, prompt_tpl


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


# ========================= LoRA / QLoRA HELPERS =========================
class LoRALinear(torch.nn.Module):
    """Drop-in LoRA wrapper for nn.Linear (or bnb Linear4bit). Base weight frozen;
    trainable lora_A/lora_B live in fp32 for a stable backward while the base stays
    in its original (4-bit / fp16) dtype."""
    def __init__(self, linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if getattr(linear, "bias", None) is not None:
            self.linear.bias.requires_grad = False
        self.scaling = alpha / r
        in_f, out_f = linear.in_features, linear.out_features
        dev = linear.weight.device
        self.lora_A = torch.nn.Linear(in_f, r, bias=False).to(dtype=torch.float32, device=dev)
        self.lora_B = torch.nn.Linear(r, out_f, bias=False).to(dtype=torch.float32, device=dev)
        self.drop   = torch.nn.Dropout(p=dropout)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(self.drop(x.to(torch.float32)))) * self.scaling
        return base + lora.to(base.dtype)


def _patch_bnb_dispatch():
    """transformers 4.44 routes bnb models through accelerate.dispatch_model, whose
    1.12 build calls model.to(device) — forbidden for 4-bit models (already placed on
    GPU during load). No-op dispatch for quantized models (same fix as PaliGemma)."""
    import transformers.modeling_utils as _mu
    if getattr(_mu, "_bnb_dispatch_patched", False):
        return
    _orig = _mu.dispatch_model
    def _safe(model, *a, **k):
        if (getattr(model, "is_loaded_in_4bit", False)
                or getattr(model, "is_loaded_in_8bit", False)
                or getattr(model, "is_quantized", False)):
            return model
        return _orig(model, *a, **k)
    _mu.dispatch_model = _safe
    _mu._bnb_dispatch_patched = True


def _relocate_buffers_to_gpu(model):
    """Dispatch was skipped → move any non-persistent buffers left on CPU to the GPU."""
    gpu = torch.device("cuda:0")
    for _, mod in model.named_modules():
        for bn, buf in list(mod._buffers.items()):
            if buf is not None and buf.device != gpu:
                mod._buffers[bn] = buf.to(gpu)


def setup_qlora_mobilevlm(model, r=16, alpha=32, dropout=0.05):
    """Freeze everything (4-bit base, vision tower, projector) and wrap LoRA on the
    MobileLLaMA decoder's attention (q/k/v/o) + MLP (gate/up/down) across all layers.
    Manual LoRALinear (not peft) keeps the model's custom generate(images=...) intact."""
    for p in model.parameters():
        p.requires_grad = False

    lm_layers = model.get_model().layers
    n_wrapped = 0
    for layer in lm_layers:
        attn = layer.self_attn
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if hasattr(attn, proj):
                setattr(attn, proj, LoRALinear(getattr(attn, proj), r, alpha, dropout))
                n_wrapped += 1
        mlp = layer.mlp
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if hasattr(mlp, proj):
                setattr(mlp, proj, LoRALinear(getattr(mlp, proj), r, alpha, dropout))
                n_wrapped += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[qlora] r={r} alpha={alpha} dropout={dropout} | wrapped {n_wrapped} Linears")
    print(f"[qlora] Trainable: {trainable/1e6:.3f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.2f}%)")
    return model


def _get_lora_state(model):
    """Best-checkpoint snapshot for early stopping — only the trainable LoRA params, on CPU."""
    return {n: p.detach().to("cpu", copy=True)
            for n, p in model.named_parameters() if p.requires_grad}


def _set_lora_state(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n].to(p.device))


# ========================= MODEL =========================
def build_mobilevlm(model_path, quantize_4bit=False):
    """Load MobileVLM V2 on a single GPU. quantize_4bit=True loads the MobileLLaMA LM
    in 4-bit NF4 (QLoRA); the vision tower + projector stay fp16."""
    disable_torch_init()
    print(f"[MobileVLM] Loading {model_path}...")

    from mobilevlm.model.mobilellama import MobileLlamaForCausalLM
    from transformers import AutoTokenizer, CLIPImageProcessor, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if quantize_4bit:
        _patch_bnb_dispatch()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("[MobileVLM] Loading LM in 4-bit NF4 (QLoRA)...")
        model = MobileLlamaForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        _relocate_buffers_to_gpu(model)
    else:
        model = MobileLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,         # no accelerate dispatch — single GPU
            low_cpu_mem_usage=True,
        )
        model = model.to(device)
    model.eval()

    vision_tower = model.get_model().get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)

    mm_projector = model.get_model().mm_projector
    mm_projector.to(device=device, dtype=torch.float16)

    image_processor = CLIPImageProcessor.from_pretrained(
        vision_tower.config._name_or_path
    )
    context_len = getattr(model.config, "max_sequence_length", 2048)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[MobileVLM] Total params: {total_params:.1f}M  context_len={context_len}")
    devices = {str(p.device) for p in model.parameters()}
    print(f"[MobileVLM] Model devices: {devices}")

    return tokenizer, model, image_processor, context_len


def match_class(generated_text, class_names):
    gen = generated_text.strip().lower().replace("_", " ")

    for i, cn in enumerate(class_names):
        if gen == cn.lower().replace("_", " "):
            return i

    best_idx, best_len = -1, 0
    for i, cn in enumerate(class_names):
        cn_clean = cn.lower().replace("_", " ")
        if cn_clean in gen and len(cn_clean) > best_len:
            best_idx, best_len = i, len(cn_clean)
    if best_idx >= 0:
        return best_idx

    for i, cn in enumerate(class_names):
        cn_clean = cn.lower().replace("_", " ")
        if gen in cn_clean and len(gen) > 2:
            return i

    return -1


@torch.no_grad()
def generate_for_image(model, tokenizer, image_processor, image, prompt_str):
    conv = conv_templates[CONV_MODE].copy()
    full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_str
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    images_tensor = process_images(
        [image], image_processor, model.config
    ).to(device, dtype=torch.float16)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # autocast(fp16) harmonizes dtypes when trainable LoRA params are fp32 but the
    # base model is 4-bit/fp16 (no-op for the un-fine-tuned baseline model).
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            temperature=1.0,
            max_new_tokens=MAX_NEW_TOKENS,
            stopping_criteria=[stopping],
            use_cache=True,
        )

    generated = tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    if stop_str and generated.endswith(stop_str):
        generated = generated[:-len(stop_str)].strip()
    return generated


def train_step_sft_mobilevlm(model, tokenizer, image_processor, image,
                             class_name, prompt_str, _debug=False):
    """SFT step. MobileLLaMA computes the loss internally when `labels` are passed;
    prepare_inputs_labels_for_multimodal expands the image token and realigns labels,
    so we just mask the prompt prefix and unmask the answer (class name + </s>)."""
    full_prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_str

    # Length of the prompt-only tokenization (assistant turn empty) — everything up to
    # here is masked; the answer that follows is supervised.
    conv_p = conv_templates[CONV_MODE].copy()
    conv_p.append_message(conv_p.roles[0], full_prompt)
    conv_p.append_message(conv_p.roles[1], None)
    n_prompt = tokenizer_image_token(
        conv_p.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").shape[0]

    # Full sequence: prompt + answer + sep2 ("</s>").
    conv_f = conv_templates[CONV_MODE].copy()
    conv_f.append_message(conv_f.roles[0], full_prompt)
    conv_f.append_message(conv_f.roles[1], class_name)
    input_ids = tokenizer_image_token(
        conv_f.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    labels = input_ids.clone()
    labels[0, :n_prompt] = IGNORE_INDEX

    images_tensor = process_images(
        [image], image_processor, model.config).to(device, dtype=torch.float16)

    if _debug:
        print(f"[DEBUG] input_ids={tuple(input_ids.shape)} n_prompt={n_prompt} "
              f"n_answer={input_ids.shape[1]-n_prompt} images={tuple(images_tensor.shape)}")

    out = model(input_ids=input_ids, images=images_tensor, labels=labels, use_cache=False)
    return out.loss


@torch.no_grad()
def quick_val_acc(model, tokenizer, image_processor, val_ds, val_indices,
                  class_names, prompt_str):
    """Greedy-generation accuracy on a small held-out subset — early-stopping signal."""
    was_training = model.training
    model.eval()
    correct = 0
    for i in val_indices:
        img, label = val_ds[i]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        if img.mode != "RGB":
            img = img.convert("RGB")
        gen = generate_for_image(model, tokenizer, image_processor, img, prompt_str)
        if match_class(gen, class_names) == label:
            correct += 1
    if was_training:
        model.train()
    return 100.0 * correct / max(len(val_indices), 1)


# ========================= EVAL LOOP =========================
@torch.no_grad()
def run_eval_generative(model, tokenizer, image_processor,
                        test_ds, class_names, prompt_tpl,
                        pwr, epoch_idx=0):
    model.eval()
    correct = total = no_match = 0
    start_t = time.time()

    class_list_str = ", ".join(cn.replace("_", " ") for cn in class_names)
    prompt_str = prompt_tpl.format(class_list_str) if "{}" in prompt_tpl else prompt_tpl

    n_samples = len(test_ds) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(test_ds))

    pbar = tqdm(range(n_samples), desc="Evaluating", leave=True)
    for step in pbar:
        _maybe_sync()
        p_start = pwr.sample_power_w()
        t0 = time.time()

        img, label = test_ds[step]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        if img.mode != "RGB":
            img = img.convert("RGB")

        generated = generate_for_image(model, tokenizer, image_processor, img, prompt_str)
        pred_idx = match_class(generated, class_names)

        if pred_idx == -1:
            no_match += 1
        if pred_idx == label:
            correct += 1
        total += 1

        _maybe_sync()
        step_t = time.time() - t0
        pwr.log_step("eval", epoch_idx, step, step_t, p_start, pwr.sample_power_w())

        acc_so_far = 100.0 * correct / total if total > 0 else 0.0
        pbar.set_postfix(
            acc=f"{acc_so_far:.1f}%",
            nomatch=no_match,
            s_per_img=f"{step_t:.2f}s",
        )

    acc = 100.0 * correct / total if total > 0 else 0.0
    elapsed = time.time() - start_t
    print(f"\nNo-match count: {no_match}/{total} ({100*no_match/total:.1f}%)")
    return acc, elapsed


@torch.no_grad()
def run_eval_hierarchical(model, tokenizer, image_processor,
                          test_ds, class_names, pwr, epoch_idx=0):
    """Two-step CIFAR-100 eval: predict superclass → predict fine class within it."""
    model.eval()
    correct = total = sc_correct = 0
    start_t = time.time()
    n_samples = len(test_ds) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(test_ds))
    pbar = tqdm(range(n_samples), desc="Evaluating (hierarchical)", leave=True)

    for step in pbar:
        _maybe_sync()
        p_start = pwr.sample_power_w()
        t0 = time.time()

        img, label = test_ds[step]
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        if img.mode != "RGB":
            img = img.convert("RGB")

        true_sc = FINE_TO_SUPERCLASS.get(class_names[label])

        # Step 1: superclass
        sc_gen  = generate_for_image(model, tokenizer, image_processor, img, SUPERCLASS_PROMPT)
        sc_pred = match_superclass(sc_gen)
        if true_sc and sc_pred == true_sc:
            sc_correct += 1

        # Step 2: fine class within predicted superclass
        fine_classes = CIFAR100_HIERARCHY[sc_pred]
        fine_gen  = generate_for_image(model, tokenizer, image_processor,
                                        img, build_fine_prompt(fine_classes))
        pred_idx  = match_fine_class(fine_gen, fine_classes, class_names)

        if pred_idx == label:
            correct += 1
        total += 1

        _maybe_sync()
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


# ========================= MAIN =========================
def run_experiment(dataset_name):
    model_tag = MODEL_PATH.split("/")[-1]
    run_tag = f"{dataset_name}/{model_tag}"
    print(f"\n{'='*80}\nSTARTING: {run_tag}\n{'='*80}")
    set_seed(SEED)

    ab_vals = [1, 2, 3, 4, 5]
    run_dir = OUTDIR / dataset_name / model_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    step_energy_gz = run_dir / "step_energy.csv.gz"
    metrics_csv    = run_dir / "epoch_metrics.csv"
    ensure_metrics_csv_header(ab_vals, metrics_csv)

    test_ds, class_names, prompt_tpl = load_test_dataset(dataset_name)
    n_samples = len(test_ds) if MAX_SAMPLES is None else min(MAX_SAMPLES, len(test_ds))
    print(f"[{dataset_name.upper()}] {len(class_names)} classes | "
          f"{n_samples} test images (of {len(test_ds)})")

    tokenizer, model, image_processor, context_len = build_mobilevlm(
        MODEL_PATH, quantize_4bit=(FT_MODE == "qlora"))
    pwr = GpuPowerMeter(device_index=GPU_INDEX, step_energy_path=step_energy_gz)

    try:
        # ---- Zero-shot / hierarchical baseline ----
        if not FT_SKIP_BASELINE:
            use_hier = USE_HIERARCHICAL and dataset_name == "cifar100"
            mode_tag = "HIERARCHICAL" if use_hier else "ZERO-SHOT"
            print(f">>> {mode_tag} evaluation...")
            pwr.reset_epoch()
            if use_hier:
                acc, eval_time = run_eval_hierarchical(
                    model, tokenizer, image_processor,
                    test_ds, class_names, pwr, epoch_idx=0)
            else:
                acc, eval_time = run_eval_generative(
                    model, tokenizer, image_processor,
                    test_ds, class_names, prompt_tpl, pwr, epoch_idx=0)

            totals = pwr.epoch_totals()
            print(f"\n[{mode_tag}] Acc={acc:.2f}% | Time={eval_time:.1f}s | "
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
        else:
            print("[SKIP BASELINE] Skipping zero-shot eval — using existing results.")

        # ---- QLoRA light-touch fine-tuning ----
        if FT_MODE == "qlora" and FT_EPOCHS > 0:
            model = setup_qlora_mobilevlm(model, r=FT_LORA_R, alpha=FT_LORA_ALPHA,
                                          dropout=FT_LORA_DROPOUT)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=FT_LR, weight_decay=0.01)

            if dataset_name == "cifar100":
                train_ds_ft = datasets.CIFAR100(root=str(DATA_ROOT), train=True,
                                                transform=None, download=True)
            else:
                train_ds_ft = datasets.DTD(root=str(DATA_ROOT), split="train",
                                           transform=None, download=True)

            # Eval prompt (long, with class list) — used for final eval, and for
            # training+validation when FT_PROMPT_MODE=="eval" (match train to eval).
            if "{}" in prompt_tpl:
                _cl = ", ".join(cn.replace("_", " ") for cn in class_names)
                eval_prompt_str = prompt_tpl.format(_cl)
            else:
                eval_prompt_str = prompt_tpl
            if FT_PROMPT_MODE == "eval":
                ft_prompt = eval_prompt_str
            else:
                ft_prompt = CIFAR100_FT_PROMPT if dataset_name == "cifar100" else DTD_FT_PROMPT

            # Disjoint held-out validation set for early stopping.
            use_earlystop = FT_VAL_EVERY > 0
            val_ds = val_indices = None
            train_pool = list(range(len(train_ds_ft)))
            if use_earlystop:
                if dataset_name == "dtd":
                    val_ds = datasets.DTD(root=str(DATA_ROOT), split="val",
                                          transform=None, download=True)
                    _vp = list(range(len(val_ds))); random.shuffle(_vp)
                    val_indices = _vp[:FT_VAL_SAMPLES]
                else:
                    random.shuffle(train_pool)
                    val_ds = train_ds_ft
                    val_indices = train_pool[:FT_VAL_SAMPLES]
                    train_pool = train_pool[FT_VAL_SAMPLES:]
                print(f"[FT] Early stopping ON: {len(val_indices)} val imgs every "
                      f"{FT_VAL_EVERY} steps, patience={FT_PATIENCE}")

            print(f"\n[FT] qlora | Epochs={FT_EPOCHS} Steps={FT_STEPS_PER_EPOCH} "
                  f"GradAccum={FT_GRAD_ACCUM} LR={FT_LR:.1e} prompt={FT_PROMPT_MODE}")
            print(f"[FT] Trainable: {sum(p.numel() for p in trainable_params)/1e6:.3f}M params")
            torch.cuda.empty_cache()

            best_val_acc = -1.0
            best_state = None
            vals_no_improve = 0
            global_step = 0
            stop_training = False
            total_train_time = 0.0
            total_train_energy = 0.0

            for epoch in range(1, FT_EPOCHS + 1):
                model.train()
                pwr.reset_epoch()
                torch.cuda.reset_peak_memory_stats(device)
                start_epoch_t = time.time()
                total_loss = 0.0
                optimizer.zero_grad(set_to_none=True)

                indices = random.sample(train_pool, min(FT_STEPS_PER_EPOCH, len(train_pool)))
                n_steps = len(indices)

                for step, idx in enumerate(tqdm(indices, desc=f"FT Epoch {epoch}", leave=False), 1):
                    img, label = train_ds_ft[idx]
                    if isinstance(img, torch.Tensor):
                        img = transforms.ToPILImage()(img)
                    elif not isinstance(img, Image.Image):
                        img = Image.fromarray(np.array(img))
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    class_name = class_names[label]
                    if step == 1 or step % 10 == 0:
                        torch.cuda.empty_cache()
                    _maybe_sync()
                    p_start = pwr.sample_power_w()
                    t0 = time.time()

                    _is_first = (epoch == 1 and step == 1)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = train_step_sft_mobilevlm(
                            model, tokenizer, image_processor, img, class_name,
                            ft_prompt, _debug=_is_first)
                    if _is_first:
                        print(f"[DEBUG] step1 loss={loss.item()}")
                    if not torch.isfinite(loss):
                        optimizer.zero_grad(set_to_none=True)
                        if step % 10 == 0 or step == 1:
                            print(f"[FT E{epoch} S{step:03d}] SKIP — loss={loss.item()}")
                        continue
                    (loss / FT_GRAD_ACCUM).backward()
                    if step % FT_GRAD_ACCUM == 0 or step == n_steps:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    step_t = time.time() - t0
                    _maybe_sync()
                    pwr.log_step("train", epoch, step, step_t, p_start, pwr.sample_power_w())
                    total_loss += loss.item()

                    if step % 10 == 0:
                        peak = torch.cuda.max_memory_allocated(device) / 1024**3
                        print(f"[FT E{epoch} S{step:03d}] loss={loss.item():.4f} "
                              f"avg={total_loss/step:.4f} peak={peak:.2f}GB")

                    # --- periodic validation + early stopping ---
                    global_step += 1
                    if use_earlystop and global_step % FT_VAL_EVERY == 0:
                        torch.cuda.empty_cache()
                        v_acc = quick_val_acc(model, tokenizer, image_processor,
                                              val_ds, val_indices, class_names, eval_prompt_str)
                        improved = v_acc > best_val_acc + 1e-6
                        if improved:
                            best_val_acc = v_acc
                            vals_no_improve = 0
                            best_state = _get_lora_state(model)
                            tag = "↑ best"
                        else:
                            vals_no_improve += 1
                            tag = f"no-improve {vals_no_improve}/{FT_PATIENCE}"
                        print(f"[FT VAL @step {global_step}] val_acc={v_acc:.2f}%  "
                              f"(best={best_val_acc:.2f}%, {tag})")
                        torch.cuda.empty_cache()
                        if vals_no_improve >= FT_PATIENCE:
                            print(f"[FT] Early stop — no val improvement for {FT_PATIENCE} "
                                  f"checks. Best val_acc={best_val_acc:.2f}%")
                            stop_training = True
                            break

                dt_train = time.time() - start_epoch_t
                ep_totals = pwr.epoch_totals()
                total_train_time   += dt_train
                total_train_energy += ep_totals["train_energy_j"]
                print(f"[FT Epoch {epoch}/{FT_EPOCHS}] avg_loss={total_loss/max(step,1):.4f} "
                      f"TrainT={dt_train/60:.1f}min TrainEnergy={ep_totals['train_energy_j']:.0f}J")
                if stop_training:
                    break

            # Restore best checkpoint (early stopping) before final eval.
            if use_earlystop and best_state is not None:
                _set_lora_state(model, best_state)
                torch.cuda.empty_cache()
                print(f"[FT] Restored best adapter (val_acc={best_val_acc:.2f}%) for final eval.")
            elif use_earlystop:
                print("[FT] No improving checkpoint captured — using last weights.")

            print("\n[FT] Running final eval...")
            pwr.reset_epoch()
            acc, eval_time = run_eval_generative(
                model, tokenizer, image_processor, test_ds, class_names, prompt_tpl,
                pwr, epoch_idx=FT_EPOCHS)
            eval_totals = pwr.epoch_totals()
            total_energy = total_train_energy + eval_totals["eval_energy_j"]
            avg_power = (total_energy / (total_train_time + eval_time)
                         if (total_train_time + eval_time) > 0 else float("nan"))
            sam_vals = compute_sam(acc, total_energy, ab_vals)
            row = [FT_EPOCHS, f"{total_train_time:.3f}", f"{eval_time:.3f}",
                   f"{total_train_time + eval_time:.3f}",
                   f"{total_train_energy:.3f}", f"{eval_totals['eval_energy_j']:.3f}",
                   f"{total_energy:.3f}", f"{avg_power:.3f}", f"{acc:.2f}"]
            for a in ab_vals:
                v = sam_vals[f"SAM_a{a}_b{a}"]
                row.append(f"{v:.6f}" if not math.isnan(v) else "nan")
            with open(metrics_csv, "a", newline="") as f:
                csv.writer(f).writerow(row)
            _best = f" | BestVal={best_val_acc:.2f}%" if use_earlystop else ""
            print(f"[FT FINAL] Acc={acc:.2f}%{_best} "
                  f"TrainT={total_train_time/60:.1f}min EvalT={eval_time/60:.1f}min")

    finally:
        pwr.close()
        del model, tokenizer, image_processor
        torch.cuda.empty_cache()

    print(f"[DONE] {run_tag} -> {metrics_csv}")
    return metrics_csv


def _parse_cli_args():
    """CLI flags override the env-var-derived defaults."""
    import argparse
    ap = argparse.ArgumentParser(
        description="MobileVLM V2-1.7B zero-shot / QLoRA on CIFAR-100 / DTD")
    ap.add_argument("--mode", choices=["zero_shot", "qlora"], default=FT_MODE)
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help="comma list, e.g. 'dtd' or 'cifar100,dtd'")
    ap.add_argument("--skip-baseline", action="store_true", default=FT_SKIP_BASELINE)
    ap.add_argument("--epochs", type=int, default=FT_EPOCHS)
    ap.add_argument("--steps", type=int, default=FT_STEPS_PER_EPOCH)
    ap.add_argument("--grad-accum", type=int, default=FT_GRAD_ACCUM)
    ap.add_argument("--lora-r", type=int, default=FT_LORA_R)
    ap.add_argument("--lora-alpha", type=int, default=FT_LORA_ALPHA)
    ap.add_argument("--lora-dropout", type=float, default=FT_LORA_DROPOUT)
    ap.add_argument("--lr", type=float, default=None,
                    help="LoRA LR (default 2e-4 for qlora, else 2e-5)")
    ap.add_argument("--ft-prompt", choices=["eval", "short"], default=FT_PROMPT_MODE)
    ap.add_argument("--val-every", type=int, default=FT_VAL_EVERY,
                    help="validate + early-stop check every N steps (0=off)")
    ap.add_argument("--val-samples", type=int, default=FT_VAL_SAMPLES)
    ap.add_argument("--patience", type=int, default=FT_PATIENCE)
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                    help="cap eval images (debug); default = full set")
    return ap.parse_args()


if __name__ == "__main__":
    _args = _parse_cli_args()
    FT_MODE            = _args.mode
    DATASETS           = [d.strip() for d in _args.datasets.split(",") if d.strip()]
    FT_SKIP_BASELINE   = _args.skip_baseline
    FT_EPOCHS          = _args.epochs
    FT_STEPS_PER_EPOCH = _args.steps
    FT_GRAD_ACCUM      = _args.grad_accum
    FT_LORA_R          = _args.lora_r
    FT_LORA_ALPHA      = _args.lora_alpha
    FT_LORA_DROPOUT    = _args.lora_dropout
    FT_PROMPT_MODE     = _args.ft_prompt
    FT_VAL_EVERY       = _args.val_every
    FT_VAL_SAMPLES     = _args.val_samples
    FT_PATIENCE        = _args.patience
    MAX_SAMPLES        = _args.max_samples
    if _args.lr is not None:
        FT_LR = _args.lr
    elif FT_MODE == "qlora":
        FT_LR = 2e-4

    print(f"[CONFIG] mode={FT_MODE} datasets={DATASETS} skip_baseline={FT_SKIP_BASELINE} "
          f"epochs={FT_EPOCHS} steps={FT_STEPS_PER_EPOCH} grad_accum={FT_GRAD_ACCUM} "
          f"lora_r={FT_LORA_R} lora_alpha={FT_LORA_ALPHA} lr={FT_LR:.1e}")
    print(f"[CONFIG] ft_prompt={FT_PROMPT_MODE} val_every={FT_VAL_EVERY} "
          f"val_samples={FT_VAL_SAMPLES} patience={FT_PATIENCE}")

    results = {}
    for ds in DATASETS:
        try:
            csv_path = run_experiment(ds)
            results[ds] = csv_path
        except Exception as e:
            print(f"\n[ERROR] {ds} failed: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()

    import pandas as pd
    for ds, csv_path in results.items():
        print(f"\n{'='*60}")
        print(f"  {ds.upper()} - MobileVLM_V2-1.7B")
        print(f"{'='*60}")
        df = pd.read_csv(csv_path)
        print(df.to_string(index=False))
        print(f"\nBest accuracy: {df['test_acc_pct'].max():.2f}%")
        print(f"Total energy:  {df['total_energy_j'].sum():.0f} J")
        print(f"Total time:    {df['total_time_s'].sum()/60:.1f} min")
