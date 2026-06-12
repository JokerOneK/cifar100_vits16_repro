"""
Runner: запускает PaLiGemma, MobileVLM, SmolVLM последовательно на DTD
с class-definition prompting. Результаты сохраняются в vlm_dtd_definitions_results/.

Usage:
    python src/run_vlms_dtd_definitions.py

Опционально только один или два VLM:
    python src/run_vlms_dtd_definitions.py --models paligemma smolvlm
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "vlm_dtd_definitions_results"

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
    env["VLM_DATASETS"]        = "dtd"
    env["VLM_USE_DEFINITIONS"] = "1"
    env["VLM_OUTDIR"]          = str(OUTDIR / model_key)
    return env


def build_cmd(model_key: str) -> list:
    script = SCRIPTS[model_key]
    cmd = [sys.executable, str(script)]
    # SmolVLM uses argparse — pass CLI flags too (env vars also work as fallback)
    if model_key == "smolvlm":
        cmd += ["--datasets", "dtd", "--use-definitions",
                "--outdir", str(OUTDIR / model_key)]
    return cmd


def run_model(model_key: str) -> int:
    name = DISPLAY_NAMES[model_key]
    print(f"\n{'='*70}")
    print(f"  [{model_key.upper()}] {name} — DTD with definitions")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(build_cmd(model_key), env=build_env(model_key))
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n[{model_key.upper()}] {status} — {elapsed/60:.1f} min")
    return result.returncode


def print_summary(results: dict):
    print(f"\n{'#'*70}")
    print("  SUMMARY — DTD class-definition prompting")
    print(f"{'#'*70}")
    print(f"  Output dir: {OUTDIR}")
    print()
    for model_key, rc in results.items():
        status = "OK" if rc == 0 else "FAILED"
        csv_path = OUTDIR / model_key / "dtd" / f"{model_key}_*" / "epoch_metrics.csv"
        # Try to find and show accuracy
        import glob
        matches = glob.glob(str(OUTDIR / model_key / "dtd" / "*" / "epoch_metrics.csv"))
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
    ap = argparse.ArgumentParser(description="Run all VLMs on DTD with definition prompts")
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
                    help="Which models to run (default: all)")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  VLM DTD DEFINITIONS RUN")
    print(f"  Models:  {args.models}")
    print(f"  Dataset: DTD (47 texture classes, 1880 test images)")
    print(f"  Prompt:  class-definition prompting")
    print(f"  Outdir:  {OUTDIR}")
    print(f"{'#'*70}")

    results = {}
    for model_key in args.models:
        results[model_key] = run_model(model_key)

    print_summary(results)


if __name__ == "__main__":
    main()
