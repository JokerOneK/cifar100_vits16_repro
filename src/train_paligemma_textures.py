"""
PaliGemma-3B Generative Zero-Shot Classification on CIFAR-100 / DTD
Adapted from paligemma.ipynb for local GPU (RTX 2060 Super 8GB).

Changes vs Kaggle notebook:
  - Paths: local instead of /kaggle/working/
  - num_workers: 0 (Windows)
  - HF token: read from env var HF_TOKEN (run `huggingface-cli login` or set env)
  - No pip install cell
  - Model reloaded per dataset (saves VRAM between runs)

VRAM note:
  PaliGemma-3B in float16 uses ~6 GB of weights alone.
  On 8 GB VRAM this is tight but workable at batch_size=1.
  If you hit OOM, install bitsandbytes and set load_in_8bit=True below.

Install:
  pip install transformers accelerate pynvml pandas

Login (required — gated model):
  huggingface-cli login
  # or: set HF_TOKEN=hf_...  in your shell
"""

import os
# NOTE: expandable_segments:True breaks on WSL2/CUDA 11.8 — causes "CUDA driver error"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "max_split_size_mb:128,garbage_collection_threshold:0.8"
)

import math, time, gzip, csv, random, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from PIL import Image
from transformers import (
    PaliGemmaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
)
import torchvision

print(f"torch={torch.__version__}  torchvision={torchvision.__version__}  "
      f"CUDA={torch.version.cuda}")

# ========================= CONFIG =========================
MODEL_ID        = "google/paligemma-3b-mix-224"
_ds_env         = os.environ.get("VLM_DATASETS", "cifar100,dtd")
DATASETS        = [d.strip() for d in _ds_env.split(",") if d.strip()]
USE_DEFINITIONS  = os.environ.get("VLM_USE_DEFINITIONS",  "0") == "1"
USE_HIERARCHICAL = os.environ.get("VLM_USE_HIERARCHICAL", "0") == "1"
EVAL_BATCH_SIZE = 1           # generative — one image at a time
SEED            = 42
MAX_NEW_TOKENS  = 16
MAX_SAMPLES     = None        # set e.g. 500 for a quick test, None = full dataset
LOAD_IN_8BIT    = False       # set True if OOM (requires bitsandbytes)
FT_MODE         = os.environ.get("VLM_FT_MODE", "zero_shot")  # "zero_shot" | "projector_lora" | "qlora"
FT_EPOCHS       = int(os.environ.get("VLM_FT_EPOCHS", "2"))
# QLoRA wants a higher LoRA LR (~2e-4) than the fp32-projector mode (~2e-5).
_default_ft_lr  = "2e-4" if FT_MODE == "qlora" else "2e-5"
FT_LR           = float(os.environ.get("VLM_FT_LR", _default_ft_lr))
FT_STEPS_PER_EPOCH = int(os.environ.get("VLM_FT_STEPS", "500"))
FT_SKIP_BASELINE = os.environ.get("VLM_FT_SKIP_BASELINE", "0") == "1"
# --- QLoRA (FT_MODE="qlora") knobs ---
FT_GRAD_ACCUM      = int(os.environ.get("VLM_FT_GRAD_ACCUM", "8"))    # effective batch = bs(1) * accum
FT_LORA_R          = int(os.environ.get("VLM_FT_LORA_R", "16"))
FT_LORA_ALPHA      = int(os.environ.get("VLM_FT_LORA_ALPHA", "32"))
FT_LORA_DROPOUT    = float(os.environ.get("VLM_FT_LORA_DROPOUT", "0.05"))
FT_TRAIN_PROJECTOR = os.environ.get("VLM_FT_TRAIN_PROJECTOR", "0") == "1"  # fp16 projector FT can NaN
# --- "light-touch" FT knobs: avoid catastrophic forgetting of the zero-shot prior ---
# Train with the SAME prompt used at eval (default), and early-stop on a held-out
# validation subset so we keep the BEST adapter instead of overfitting to loss=0.
FT_PROMPT_MODE = os.environ.get("VLM_FT_PROMPT", "eval")   # "eval" (match eval) | "short"
FT_VAL_EVERY   = int(os.environ.get("VLM_FT_VAL_EVERY", "25"))    # validate every N steps (0=off)
FT_VAL_SAMPLES = int(os.environ.get("VLM_FT_VAL_SAMPLES", "120")) # held-out images for val
FT_PATIENCE    = int(os.environ.get("VLM_FT_PATIENCE", "3"))      # stop after N vals w/o improvement
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
_outdir_env = os.environ.get("VLM_OUTDIR")
OUTDIR     = Path(_outdir_env) if _outdir_env else SCRIPT_DIR / "paligemma_results"
DATA_ROOT  = SCRIPT_DIR / "data"
GPU_INDEX  = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram_gb:.1f} GB")
    if vram_gb < 8.5:
        print("[WARN] <8.5 GB VRAM — PaliGemma-3B float16 may be tight. "
              "Set LOAD_IN_8BIT=True if OOM.")


# ========================= HF LOGIN =========================
def _hf_login():
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("[HF] Logged in via HF_TOKEN env var.")
    else:
        print("[HF] No HF_TOKEN env var found.")
        print("     Make sure you ran: huggingface-cli login")

_hf_login()


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

# Open-ended prompts — no class list to avoid exceeding PaliGemma's sequence limit.
# match_class() handles fuzzy matching against class_names after generation.
CIFAR100_PROMPT = (
    "Classify this image into exactly one of these categories: {}. "
    "Answer with ONLY the category name, nothing else."
)
DTD_PROMPT = (
    "What texture or pattern does this image show? Choose exactly one from: {}. "
    "Answer with ONLY the texture name, nothing else."
)

# Short prompts for FT — class list is too long (100 classes ≈ 320 tokens) and
# pushes fp32 logits over 8GB VRAM during backward. The model just needs to learn
# image → class name; the verbose instruction is only useful at eval time for matching.
CIFAR100_FT_PROMPT = "What is in this image?"
DTD_FT_PROMPT = "What texture is shown?"

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
            print(f"[WARN] Hardcoded class list mismatch! Using ds.classes.")
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


# ========================= MODEL =========================
def build_paligemma(model_id, quantize_4bit=False):
    print(f"[PaliGemma] Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)

    if quantize_4bit:
        # --- Version-skew workaround (transformers 4.44 + accelerate 1.12) ---
        # transformers 4.44 still routes bnb models through accelerate.dispatch_model,
        # whose 1.12 build calls model.to(device) — forbidden for 4-bit models (they're
        # already placed on GPU during load). Newer transformers skip dispatch for
        # quantized models; we replicate that by no-op'ing dispatch for them.
        import transformers.modeling_utils as _mu
        if not getattr(_mu, "_bnb_dispatch_patched", False):
            _orig_dispatch = _mu.dispatch_model
            def _safe_dispatch(model, *a, **k):
                if (getattr(model, "is_loaded_in_4bit", False)
                        or getattr(model, "is_loaded_in_8bit", False)
                        or getattr(model, "is_quantized", False)):
                    return model  # already on the correct device, nothing to dispatch
                return _orig_dispatch(model, *a, **k)
            _mu.dispatch_model = _safe_dispatch
            _mu._bnb_dispatch_patched = True

        # NF4 4-bit base for QLoRA: ~2 GB instead of ~6 GB fp16, freeing room on
        # 8 GB for LoRA adapters + optimizer state + backward activations.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("[PaliGemma] Loading in 4-bit NF4 (QLoRA)...")
        # quantized weights are placed by device_map — do NOT .to(device) afterwards.
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        # Skipping dispatch (above) means non-persistent buffers such as SigLIP's
        # `position_ids` were left on CPU while all weights are on GPU. Relocate them.
        _gpu = torch.device("cuda:0")
        for _, _mod in model.named_modules():
            for _bn, _buf in list(_mod._buffers.items()):
                if _buf is not None and _buf.device != _gpu:
                    _mod._buffers[_bn] = _buf.to(_gpu)
    else:
        load_kwargs = dict(
            torch_dtype=torch.float16,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        if LOAD_IN_8BIT:
            load_kwargs["load_in_8bit"] = True
            load_kwargs.pop("torch_dtype")  # incompatible with 8-bit
            print("[PaliGemma] Loading in 8-bit (bitsandbytes)...")

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs
        ).to(device)
    # Disable KV-cache: it keeps the autograd graph alive across forward passes
    # and inflates backward memory (root cause of "CUDA unknown error" on 8GB).
    model.config.use_cache = False
    if hasattr(model, "language_model"):
        model.language_model.config.use_cache = False
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[PaliGemma] Total params: {total_params:.1f}M")
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        print(f"[PaliGemma] VRAM after load: {used:.2f} GB")

    return model, processor


# ========================= LoRA HELPERS =========================
class LoRALinear(torch.nn.Module):
    """Drop-in LoRA wrapper for nn.Linear. Keeps original weight frozen.
    Trainable params (lora_A, lora_B) live in fp32 for stable backward,
    while the frozen base linear stays in the model's original dtype (fp16)."""
    def __init__(self, linear: torch.nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.05):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if linear.bias is not None:
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


def setup_projector_lora_paligemma(model, n_lm_layers: int = 4, lora_r: int = 4):
    """Projector full FT + LoRA r=lora_r on first n_lm_layers of Gemma LM. All else frozen."""
    for p in model.parameters():
        p.requires_grad = False

    lm_layers = model.language_model.model.layers
    for i in range(min(n_lm_layers, len(lm_layers))):
        attn = lm_layers[i].self_attn
        for proj_name in ("q_proj", "v_proj"):
            setattr(attn, proj_name,
                    LoRALinear(getattr(attn, proj_name), r=lora_r, alpha=lora_r * 2))

    # Cast multi_modal_projector to fp32 — training a fp16 projector via AdamW
    # produces NaN gradients. Eval still works because model.generate() is wrapped
    # in autocast(fp16) which harmonizes dtypes between fp32 projector and fp16 LM.
    for name, p in model.named_parameters():
        if "multi_modal_projector" in name:
            p.data = p.data.to(torch.float32)
            p.requires_grad = True

    # Gradient checkpointing — re-compute activations during backward to cut peak VRAM.
    # use_reentrant=True is more stable on transformers 4.41 / torch 2.4 than the
    # newer non-reentrant variant which has known NaN issues with PaLiGemma forward.
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("[projector_lora] Gradient checkpointing enabled (use_reentrant=True)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[projector_lora] Trainable: {trainable/1e6:.3f}M / {total/1e6:.1f}M "
          f"({100*trainable/total:.2f}%)")
    return model


def setup_qlora_paligemma(model, r=16, alpha=32, dropout=0.05, train_projector=False):
    """QLoRA: 4-bit frozen base (loaded via build_paligemma(quantize_4bit=True)) +
    LoRA adapters on the Gemma language model (attention q/k/v/o + MLP gate/up/down,
    all layers). Uses the `peft` library, which integrates with bitsandbytes Linear4bit.

    Compared to setup_projector_lora_paligemma (r=4, q/v on 4 layers, fp32 projector),
    this adapts far more of the model so fine-tuning can actually move accuracy — the
    4-bit base buys back the VRAM that the extra adapters cost.
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Casts layernorms to fp32, enables input grads, turns off use_cache, and wires
    # gradient checkpointing — all required for stable 4-bit backward.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Regex restricted to the Gemma LM so we DON'T also adapt the SigLIP vision tower
    # (whose attention also exposes q_proj/k_proj/v_proj). peft re.fullmatch's this.
    target = (r"language_model\..*\."
              r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    # Training the (non-quantized, fp16) projector via AdamW is prone to NaN — off by
    # default; enable with VLM_FT_TRAIN_PROJECTOR=1 if you want to risk it.
    modules_to_save = ["multi_modal_projector"] if train_projector else None

    lora_cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=target,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_cfg)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    _register_logits_keep_hook(model)

    print(f"[qlora] r={r} alpha={alpha} dropout={dropout} "
          f"train_projector={train_projector}")
    model.print_trainable_parameters()
    return model


# When set to an int K, the lm_head pre-hook (see _register_logits_keep_hook) computes
# logits for only the LAST K sequence positions. This caps the full-vocab fp32 logits
# tensor — for CIFAR's 495-token prompt the full (1,495,256000) fp32 tensor is ~0.5 GB,
# a large contiguous alloc that intermittently fails as "CUDA unknown error" on 8 GB
# WSL2. We only supervise the last few (answer) tokens, so the rest are never needed.
_LOGITS_KEEP = None


def _register_logits_keep_hook(model):
    """Cap lm_head to the last `_LOGITS_KEEP` positions during training. None = full
    (used at eval/generation, which needs logits for the running last token only)."""
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        for name, mod in model.named_modules():
            if name.endswith("lm_head") and isinstance(mod, torch.nn.Linear):
                lm_head = mod
                break
    if lm_head is None:
        print("[qlora] WARN: lm_head not found — logits-keep memory cap disabled.")
        return

    def _pre_hook(module, args):
        k = _LOGITS_KEEP
        if k is None or not args:
            return None
        hs = args[0]
        if torch.is_tensor(hs) and hs.dim() == 3 and hs.shape[1] > k:
            return (hs[:, -k:, :],) + tuple(args[1:])
        return None

    lm_head.register_forward_pre_hook(_pre_hook)
    print("[qlora] lm_head logits-keep hook registered (caps fp32 logits memory).")


def train_step_sft_paligemma(model, processor, image, class_name, prompt_str, _debug=False):
    """SFT step: compute cross-entropy loss on class-name tokens.

    Computes loss manually from logits because older transformers versions
    (4.41-4.42) do not compute loss inside PaliGemmaForConditionalGeneration.
    """
    global _LOGITS_KEEP
    full_text = f"{prompt_str} {class_name}"
    inputs = processor(images=image, text=full_text, return_tensors="pt").to(device)
    # Cast pixel_values to fp16 to match the frozen vision_tower's weights.
    # Don't use next(model.parameters()).dtype — it now returns fp32 (projector).
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

    if _debug:
        for k, v in inputs.items():
            if torch.is_tensor(v):
                nan_str = (f"has_nan={v.isnan().any().item()}" if v.is_floating_point()
                           else "is_int")
                print(f"[DEBUG] input[{k}] shape={tuple(v.shape)} dtype={v.dtype} {nan_str}")

    # We supervise the last n_unmask tokens (class name + trailing EOS). Tell the
    # lm_head pre-hook to only compute logits for the last (n_unmask+1) positions —
    # everything earlier is irrelevant to the loss and just wastes ~0.5 GB of VRAM.
    answer_ids = processor.tokenizer(class_name, add_special_tokens=False,
                                      return_tensors="pt").input_ids
    n_answer = max(answer_ids.shape[1], 1)
    seq = inputs["input_ids"].shape[1]
    n_unmask = min(n_answer + 1, seq)
    keep = min(n_unmask + 1, seq)

    _LOGITS_KEEP = keep
    try:
        outputs = model(**inputs)
        logits = outputs.logits  # (1, L, vocab); L == keep when the hook fires, else seq
    finally:
        _LOGITS_KEEP = None

    if _debug:
        print(f"[DEBUG] logits dtype={logits.dtype} shape={tuple(logits.shape)} "
              f"has_nan={logits.isnan().any().item()} has_inf={logits.isinf().any().item()} "
              f"min={logits.min().item():.3f} max={logits.max().item():.3f}")

    # Causal-LM shift: the logit at sequence position p predicts token p+1. logits
    # covers the last L positions [seq-L, seq-1], so sliced index i ↔ global (seq-L+i).
    # The n_unmask supervised tokens sit at global [seq-n_unmask, seq-1] and are
    # predicted by logit indices [L-n_unmask-1, L-2]. (When L==seq this reduces to the
    # usual full-sequence shift.)
    L = logits.shape[1]
    relevant_logits = logits[0, L - n_unmask - 1: L - 1, :]      # (n_unmask, vocab)
    relevant_labels = inputs["input_ids"][0, seq - n_unmask: seq]
    if relevant_logits.dtype != torch.float32:
        relevant_logits = relevant_logits.float()
    return torch.nn.functional.cross_entropy(relevant_logits, relevant_labels)


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
def generate_for_image(model, processor, image, prompt_str):
    # Do NOT include "<image>" in text — transformers 4.38+ processor adds
    # 256 image tokens automatically. Adding it manually gives 257 positions
    # for 256 features → masked_scatter assertion on PyTorch < 2.6.
    inputs = processor(
        images=image, text=prompt_str,
        return_tensors="pt",
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    # autocast(fp16) harmonizes dtypes — needed when trainable params (projector,
    # LoRA A/B) are fp32 but the rest of the model is fp16.
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    generated = processor.decode(
        output_ids[0, input_len:], skip_special_tokens=True
    ).strip()
    return generated


# ========================= EVAL LOOP =========================
@torch.no_grad()
def run_eval_generative(model, processor, test_ds, class_names,
                        prompt_tpl, pwr, epoch_idx=0):
    model.eval()
    correct = total = no_match = 0
    start_t = time.time()

    # Format prompt with the class list (the {} placeholder in CIFAR100_PROMPT / DTD_PROMPT).
    # Without the explicit list, the model answers free-form and match_class fails 100%.
    if "{}" in prompt_tpl:
        class_list_str = ", ".join(cn.replace("_", " ") for cn in class_names)
        prompt_str = prompt_tpl.format(class_list_str)
    else:
        prompt_str = prompt_tpl

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

        generated = generate_for_image(model, processor, img, prompt_str)
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
def run_eval_hierarchical(model, processor, test_ds, class_names, pwr, epoch_idx=0):
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
        sc_gen  = generate_for_image(model, processor, img, SUPERCLASS_PROMPT)
        sc_pred = match_superclass(sc_gen)
        if true_sc and sc_pred == true_sc:
            sc_correct += 1

        # Step 2: fine class within predicted superclass
        fine_classes = CIFAR100_HIERARCHY[sc_pred]
        fine_gen  = generate_for_image(model, processor, img, build_fine_prompt(fine_classes))
        pred_idx  = match_fine_class(fine_gen, fine_classes, class_names)

        if pred_idx == label:
            correct += 1
        total += 1

        _maybe_sync()
        step_t = time.time() - t0
        pwr.log_step("eval", epoch_idx, step, step_t, p_start, pwr.sample_power_w())

        acc_so_far = 100.0 * correct   / total if total > 0 else 0.0
        sc_acc     = 100.0 * sc_correct / total if total > 0 else 0.0
        pbar.set_postfix(acc=f"{acc_so_far:.1f}%", sc=f"{sc_acc:.1f}%",
                         s_img=f"{step_t:.1f}s")

    acc    = 100.0 * correct    / total if total > 0 else 0.0
    sc_acc = 100.0 * sc_correct / total if total > 0 else 0.0
    elapsed = time.time() - start_t
    print(f"\nSuperclass acc: {sc_acc:.2f}%  |  Final acc: {acc:.2f}%")
    return acc, elapsed


@torch.no_grad()
def quick_val_acc(model, processor, val_ds, val_indices, class_names, prompt_str):
    """Greedy-generation accuracy on a small held-out subset — the early-stopping
    signal. Uses the SAME prompt as final eval so train/val/eval are consistent."""
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
        gen = generate_for_image(model, processor, img, prompt_str)
        if match_class(gen, class_names) == label:
            correct += 1
    if was_training:
        model.train()
    return 100.0 * correct / max(len(val_indices), 1)


# ========================= MAIN =========================
def run_experiment(dataset_name):
    model_tag = MODEL_ID.split("/")[-1]
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

    # Load model fresh each dataset to avoid VRAM fragmentation.
    # QLoRA needs the 4-bit base; baseline eval (if run) then uses the 4-bit model too.
    model, processor = build_paligemma(MODEL_ID, quantize_4bit=(FT_MODE == "qlora"))
    pwr = GpuPowerMeter(device_index=GPU_INDEX, step_energy_path=step_energy_gz)

    try:
        if not FT_SKIP_BASELINE:
            use_hier = USE_HIERARCHICAL and dataset_name == "cifar100"
            mode_tag = "HIERARCHICAL" if use_hier else "ZERO-SHOT"
            print(f">>> {mode_tag} evaluation...")
            pwr.reset_epoch()
            if use_hier:
                acc, eval_time = run_eval_hierarchical(
                    model, processor, test_ds, class_names, pwr, epoch_idx=0)
            else:
                acc, eval_time = run_eval_generative(
                    model, processor, test_ds, class_names, prompt_tpl, pwr, epoch_idx=0)

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
            print(f"[SKIP BASELINE] Skipping zero-shot eval — using existing results.")

        # --- SFT fine-tuning (projector_lora / qlora modes) ---
        if FT_MODE in ("projector_lora", "qlora") and FT_EPOCHS > 0:
            if FT_MODE == "qlora":
                model = setup_qlora_paligemma(
                    model, r=FT_LORA_R, alpha=FT_LORA_ALPHA,
                    dropout=FT_LORA_DROPOUT, train_projector=FT_TRAIN_PROJECTOR)
            else:
                model = setup_projector_lora_paligemma(model, n_lm_layers=4, lora_r=4)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=FT_LR, weight_decay=0.01)

            if dataset_name == "cifar100":
                train_ds_ft = datasets.CIFAR100(root=str(DATA_ROOT), train=True,
                                                transform=None, download=True)
            else:
                train_ds_ft = datasets.DTD(root=str(DATA_ROOT), split="train",
                                           transform=None, download=True)

            # Build the eval prompt (long, with class list) once. When FT_PROMPT_MODE
            # =="eval" we train + validate with this SAME prompt, so the model is
            # fine-tuned in exactly the format it's scored in (fixes train/eval mismatch).
            if "{}" in prompt_tpl:
                _class_list = ", ".join(cn.replace("_", " ") for cn in class_names)
                eval_prompt_str = prompt_tpl.format(_class_list)
            else:
                eval_prompt_str = prompt_tpl
            if FT_PROMPT_MODE == "eval":
                ft_prompt = eval_prompt_str
            else:
                ft_prompt = CIFAR100_FT_PROMPT if dataset_name == "cifar100" else DTD_FT_PROMPT

            # Carve a disjoint validation subset for early stopping (qlora/peft only).
            use_earlystop = FT_VAL_EVERY > 0 and FT_MODE == "qlora"
            val_ds = val_indices = None
            train_pool = list(range(len(train_ds_ft)))
            if use_earlystop:
                if dataset_name == "dtd":
                    val_ds = datasets.DTD(root=str(DATA_ROOT), split="val",
                                          transform=None, download=True)
                    _vp = list(range(len(val_ds))); random.shuffle(_vp)
                    val_indices = _vp[:FT_VAL_SAMPLES]
                else:
                    # CIFAR has no val split — hold out a disjoint slice of train.
                    random.shuffle(train_pool)
                    val_ds = train_ds_ft
                    val_indices = train_pool[:FT_VAL_SAMPLES]
                    train_pool = train_pool[FT_VAL_SAMPLES:]
                print(f"[FT] Early stopping ON: {len(val_indices)} val imgs every "
                      f"{FT_VAL_EVERY} steps, patience={FT_PATIENCE}")

            print(f"\n[FT] {FT_MODE} | Epochs={FT_EPOCHS} Steps={FT_STEPS_PER_EPOCH} "
                  f"GradAccum={FT_GRAD_ACCUM} LR={FT_LR:.1e} prompt={FT_PROMPT_MODE}")
            print(f"[FT] Trainable: {sum(p.numel() for p in trainable_params)/1e6:.3f}M params")

            torch.cuda.empty_cache()

            # Early-stopping bookkeeping. Best adapter kept on CPU (~80 MB) and restored
            # before final eval, so we report the BEST checkpoint, not the overfit one.
            try:
                from peft import get_peft_model_state_dict, set_peft_model_state_dict
            except Exception:
                get_peft_model_state_dict = set_peft_model_state_dict = None
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
                optimizer.zero_grad(set_to_none=True)  # fresh accumulator each epoch

                indices = random.sample(train_pool,
                                        min(FT_STEPS_PER_EPOCH, len(train_pool)))
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
                    # autocast(fp16) needed: trainable LoRA + projector are fp32 but
                    # base model is fp16 — autocast harmonizes the dtype mismatch in
                    # matmuls. pixel_values cast to fp16 explicitly inside train_step.
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = train_step_sft_paligemma(
                            model, processor, img, class_name, ft_prompt, _debug=_is_first)
                    if _is_first:
                        print(f"[DEBUG] step1 loss={loss.item()}")
                    if not torch.isfinite(loss):
                        # Drop whatever grads accumulated so far this window — a NaN/Inf
                        # loss would corrupt trainable params on the next optimizer.step().
                        optimizer.zero_grad(set_to_none=True)
                        if step % 10 == 0 or step == 1:
                            print(f"[FT E{epoch} S{step:03d}] SKIP — loss={loss.item()}")
                        continue
                    # Gradient accumulation: scale so the summed grads ≈ mean over the window.
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
                        # Free training cache before validation generation — on 8 GB
                        # WSL2 the train+generate memory mix fragments VRAM and surfaces
                        # as "CUDA unknown error" on the next backward (esp. CIFAR's
                        # longer ~495-token prompt). Defragment before and after.
                        torch.cuda.empty_cache()
                        v_acc = quick_val_acc(model, processor, val_ds, val_indices,
                                              class_names, eval_prompt_str)
                        improved = v_acc > best_val_acc + 1e-6
                        if improved:
                            best_val_acc = v_acc
                            vals_no_improve = 0
                            if get_peft_model_state_dict is not None:
                                # Keep the best adapter on CPU — avoids a second ~78 MB
                                # copy living on the GPU and the fragmentation it causes.
                                best_state = {k: v.detach().to("cpu", copy=True)
                                              for k, v in get_peft_model_state_dict(model).items()}
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
                      f"TrainT={dt_train/60:.1f}min "
                      f"TrainEnergy={ep_totals['train_energy_j']:.0f}J")

                if stop_training:
                    break

            # ---- Restore best adapter (early stopping) before final eval ----
            if use_earlystop and best_state is not None and set_peft_model_state_dict is not None:
                set_peft_model_state_dict(model, best_state)  # loads CPU tensors into GPU params
                torch.cuda.empty_cache()
                print(f"[FT] Restored best adapter (val_acc={best_val_acc:.2f}%) for final eval.")
            elif use_earlystop:
                print(f"[FT] No improving checkpoint captured — using last weights.")

            # ---- Final eval (once, after all epochs) ----
            # Use the LONG prompt with class list for eval (helps match_class).
            # Eval is no-grad so memory is much lower — long prompt fits fine.
            print(f"\n[FT] All {FT_EPOCHS} epochs done — running final eval...")
            pwr.reset_epoch()
            acc, eval_time = run_eval_generative(
                model, processor, test_ds, class_names, prompt_tpl, pwr,
                epoch_idx=FT_EPOCHS)

            eval_totals = pwr.epoch_totals()
            total_energy = total_train_energy + eval_totals["eval_energy_j"]
            avg_power = (total_energy / (total_train_time + eval_time)
                         if (total_train_time + eval_time) > 0 else float("nan"))
            sam_vals = compute_sam(acc, total_energy, ab_vals)
            row = [FT_EPOCHS, f"{total_train_time:.3f}", f"{eval_time:.3f}",
                   f"{total_train_time + eval_time:.3f}",
                   f"{total_train_energy:.3f}", f"{eval_totals['eval_energy_j']:.3f}",
                   f"{total_energy:.3f}", f"{avg_power:.3f}",
                   f"{acc:.2f}"]
            for a in ab_vals:
                v = sam_vals[f"SAM_a{a}_b{a}"]
                row.append(f"{v:.6f}" if not math.isnan(v) else "nan")
            with open(metrics_csv, "a", newline="") as f:
                csv.writer(f).writerow(row)

            _best_str = f" | BestVal={best_val_acc:.2f}%" if use_earlystop else ""
            print(f"[FT FINAL] Acc={acc:.2f}%{_best_str} "
                  f"TrainT={total_train_time/60:.1f}min EvalT={eval_time/60:.1f}min")

    finally:
        pwr.close()
        del model, processor
        torch.cuda.empty_cache()

    print(f"[DONE] {run_tag} -> {metrics_csv}")
    return metrics_csv


def _parse_cli_args():
    """CLI flags override the env-var-derived defaults. Env workflow still works:
    unset flags fall back to the VLM_* defaults computed at import time."""
    import argparse
    ap = argparse.ArgumentParser(
        description="PaliGemma-3B zero-shot / QLoRA on CIFAR-100 / DTD")
    ap.add_argument("--mode", choices=["zero_shot", "projector_lora", "qlora"],
                    default=FT_MODE, help="run mode (default from VLM_FT_MODE)")
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help="comma list, e.g. 'dtd' or 'cifar100,dtd'")
    ap.add_argument("--skip-baseline", action="store_true", default=FT_SKIP_BASELINE,
                    help="skip the zero-shot baseline eval before FT")
    ap.add_argument("--epochs", type=int, default=FT_EPOCHS)
    ap.add_argument("--steps", type=int, default=FT_STEPS_PER_EPOCH,
                    help="train samples per epoch")
    ap.add_argument("--grad-accum", type=int, default=FT_GRAD_ACCUM)
    ap.add_argument("--lora-r", type=int, default=FT_LORA_R)
    ap.add_argument("--lora-alpha", type=int, default=FT_LORA_ALPHA)
    ap.add_argument("--lora-dropout", type=float, default=FT_LORA_DROPOUT)
    ap.add_argument("--lr", type=float, default=None,
                    help="LoRA LR (default 2e-4 for qlora, else 2e-5)")
    ap.add_argument("--train-projector", action="store_true", default=FT_TRAIN_PROJECTOR)
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                    help="cap eval images (debug); default = full set")
    ap.add_argument("--ft-prompt", choices=["eval", "short"], default=FT_PROMPT_MODE,
                    help="train prompt: 'eval' matches the scoring prompt (recommended)")
    ap.add_argument("--val-every", type=int, default=FT_VAL_EVERY,
                    help="validate + early-stop check every N steps (0=off)")
    ap.add_argument("--val-samples", type=int, default=FT_VAL_SAMPLES,
                    help="held-out images used for validation")
    ap.add_argument("--patience", type=int, default=FT_PATIENCE,
                    help="stop after N validations without improvement")
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
    FT_TRAIN_PROJECTOR = _args.train_projector
    MAX_SAMPLES        = _args.max_samples
    FT_PROMPT_MODE     = _args.ft_prompt
    FT_VAL_EVERY       = _args.val_every
    FT_VAL_SAMPLES     = _args.val_samples
    FT_PATIENCE        = _args.patience
    if _args.lr is not None:
        FT_LR = _args.lr
    elif FT_MODE == "qlora":
        FT_LR = 2e-4   # re-derive: env default was for the import-time mode

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
        print(f"  {ds.upper()} - PaliGemma-3B-mix-224")
        print(f"{'='*60}")
        df = pd.read_csv(csv_path)
        print(df.to_string(index=False))
        print(f"\nBest accuracy: {df['test_acc_pct'].max():.2f}%")
        print(f"Total energy:  {df['total_energy_j'].sum():.0f} J")
        print(f"Total time:    {df['total_time_s'].sum()/60:.1f} min")
