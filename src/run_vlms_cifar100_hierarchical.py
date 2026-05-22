"""
Runner: запускает PaLiGemma, MobileVLM, SmolVLM последовательно на CIFAR-100
с двухшаговым иерархическим prompting.

Шаг 1: предсказать суперкласс (20 классов)
Шаг 2: предсказать финальный класс внутри суперкласса (5 классов)

Результаты сохраняются в vlm_cifar100_hierarchical_results/.

Usage:
    python src/run_vlms_cifar100_hierarchical.py

Только один или два VLM:
    python src/run_vlms_cifar100_hierarchical.py --models smolvlm
    python src/run_vlms_cifar100_hierarchical.py --models paligemma mobilevlm
"""

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "vlm_cifar100_hierarchical_results"

ALL_MODELS = ["paligemma", "mobilevlm", "smolvlm"]

SCRIPTS = {
    "paligemma": SCRIPT_DIR / "train_paligemma_textures.py",
    "mobilevlm": SCRIPT_DIR / "train_mobilevlm_textures.py",
    "smolvlm":   SCRIPT_DIR / "train_smolvlm_textures.py",
}

DISPLAY_NAMES = {
    "paligemma": "PaLiGemma-3B",
    "mobilevlm": "MobileVLM V2-1.7B",
    "smolvlm":   "SmolVLM-256M",
}


def build_env(model_key: str) -> dict:
    env = os.environ.copy()
    env["VLM_DATASETS"]        = "cifar100"
    env["VLM_USE_HIERARCHICAL"] = "1"
    env["VLM_OUTDIR"]          = str(OUTDIR / model_key)
    return env


def build_cmd(model_key: str) -> list:
    script = SCRIPTS[model_key]
    cmd = [sys.executable, str(script)]
    if model_key == "smolvlm":
        cmd += ["--datasets", "cifar100", "--use-hierarchical",
                "--outdir", str(OUTDIR / model_key)]
    return cmd


def run_model(model_key: str) -> int:
    name = DISPLAY_NAMES[model_key]
    print(f"\n{'='*70}")
    print(f"  [{model_key.upper()}] {name} — CIFAR-100 hierarchical")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(build_cmd(model_key), env=build_env(model_key))
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[{model_key.upper()}] {status} — {elapsed/60:.1f} min")
    return result.returncode


def print_summary(results: dict):
    print(f"\n{'#'*70}")
    print("  SUMMARY — CIFAR-100 hierarchical prompting")
    print(f"{'#'*70}")
    print(f"  Output dir: {OUTDIR}")
    print()
    for model_key, rc in results.items():
        status = "OK" if rc == 0 else "FAILED"
        matches = glob.glob(str(OUTDIR / model_key / "cifar100" / "*" / "epoch_metrics.csv"))
        acc_str = ""
        if matches:
            try:
                import csv as _csv
                with open(matches[0]) as f:
                    rows = list(_csv.DictReader(f))
                if rows:
                    acc_str = f"  acc={float(rows[-1]['test_acc_pct']):.2f}%"
            except Exception:
                pass
        print(f"  {DISPLAY_NAMES[model_key]:20s}  {status}{acc_str}")
    print(f"{'#'*70}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Run VLMs on CIFAR-100 with two-step hierarchical prompting")
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
                    help="Which models to run (default: all)")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  VLM CIFAR-100 HIERARCHICAL RUN")
    print(f"  Models:  {args.models}")
    print(f"  Dataset: CIFAR-100 (100 classes → 20 superclasses × 5 fine)")
    print(f"  Prompt:  Step 1 = superclass (20), Step 2 = fine class (5)")
    print(f"  Outdir:  {OUTDIR}")
    print(f"{'#'*70}")

    results = {}
    for model_key in args.models:
        results[model_key] = run_model(model_key)

    print_summary(results)


if __name__ == "__main__":
    main()
