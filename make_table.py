import os, pandas as pd
from metrics import compute_sam, auto_scale_for_baseline

MODES = ['ft','bitfit','lora8','lora4']

def load_all(results_dir='.'):
    dfs = {}
    for m in MODES:
        path = os.path.join(results_dir, f"results_{m}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['mode'] = m
            dfs[m] = df
    return dfs

def summarize(dfs):
    rows = []
    for m, df in dfs.items():
        row = df.mean(numeric_only=True).to_dict()
        row['mode'] = m
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=".")
    ap.add_argument("--formula", default="acc_over_log_energy",
                    choices=["acc_over_log_energy","acc_over_energy","acc_over_power","acc_over_time","acc_over_energy_mem"])
    ap.add_argument("--electricity_col", default="energy_Wh", help="Column that represents 'electricity' in the SAM formula")
    ap.add_argument("--baseline_mode", default="ft")
    ap.add_argument("--sam1_target", type=float, default=None,
                    help="If set, calibrate scale so that SAM1 of baseline equals this number (applies only to formulas != acc_over_log_energy).")
    args = ap.parse_args()

    dfs = load_all(args.results_dir)
    if not dfs:
        raise SystemExit("No results_*.csv files found. Run training first.")

    summary = summarize(dfs).set_index('mode')

    # compute SAM1..SAM5 (alpha=beta=k)
    for k in range(1, 6):
        summary[f"SAM{k}"] = compute_sam(summary, alpha=k, beta=k, formula=args.formula, electricity_col=args.electricity_col)

    # Calibration only makes sense for scale-based formulas; for log-energy variant we keep exact definition
    if args.sam1_target is not None and args.formula != "acc_over_log_energy" and args.baseline_mode in summary.index:
        baseline_value = summary.loc[args.baseline_mode, "SAM1"]
        scale = auto_scale_for_baseline(baseline_value, args.sam1_target)
        for k in range(1, 6):
            summary[f"SAM{k}"] *= scale

    cols = ["test_acc","hours","avg_watts","energy_Wh","avg_mem_GB","peak_mem_GB",
            "avg_gpu_util_pct","avg_mem_util_pct","SAM1","SAM2","SAM3","SAM4","SAM5"]
    summary = summary.reset_index()[["mode"] + cols]

    out = os.path.join(args.results_dir, "summary_with_SAM.csv")
    summary.to_csv(out, index=False)
    print(summary)
    print(f"Saved -> {out}")
