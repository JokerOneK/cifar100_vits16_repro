
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def read_csv_maybe_gz(p):
    p = Path(p)
    if str(p).endswith(".gz"):
        return pd.read_csv(p, compression="gzip")
    return pd.read_csv(p)

def steps_per_epoch(df_layer_raw):
    g = df_layer_raw.groupby(["epoch"])["step"].max()
    return int(g.iloc[0])

def peak_mem_per_step(mem_raw):
    d = (mem_raw.groupby(["epoch","step"])["mem_mib"].max()
         .reset_index())
    # global step assumes uniform steps per epoch; uses per-epoch max
    d["global_step"] = (d["epoch"]-1)*d["step"].max() + d["step"]
    return d

def step_time_from_layer(layer_raw):
    step = (layer_raw.groupby(["epoch","step"])["ms"].sum().reset_index())
    step["global_step"] = (step["epoch"]-1)*step["step"].max() + step["step"]
    return step

def fwd_counts_per_step(layer_raw):
    c = (layer_raw[layer_raw["phase"]=="fwd"]
         .groupby(["epoch","step"]).size().reset_index(name="fwd_events"))
    c["global_step"] = (c["epoch"]-1)*c["step"].max() + c["step"]
    return c

def phase_sequence_for_step(layer_raw, epoch=1, step=1):
    df = layer_raw[(layer_raw["epoch"]==epoch) & (layer_raw["step"]==step)].copy()
    df = df.reset_index(drop=True)
    df["order"] = df.index
    phase_map = {"fwd":1, "bwd":0}
    df["phase_val"] = df["phase"].map(phase_map)
    return df[["order","phase","phase_val","layer"]]

def main(args):
    base = Path(args.outdir)
    base.mkdir(parents=True, exist_ok=True)

    no_mem_raw = read_csv_maybe_gz(args.no_ckpt_mem_raw)
    no_mem_epoch = read_csv_maybe_gz(args.no_ckpt_mem_epoch)
    no_layer_raw = read_csv_maybe_gz(args.no_ckpt_layer_raw)
    no_layer_epoch = read_csv_maybe_gz(args.no_ckpt_layer_epoch)

    ck_mem_raw = read_csv_maybe_gz(args.ckpt_mem_raw)
    ck_mem_epoch = read_csv_maybe_gz(args.ckpt_mem_epoch)
    ck_layer_raw = read_csv_maybe_gz(args.ckpt_layer_raw)
    ck_layer_epoch = read_csv_maybe_gz(args.ckpt_layer_epoch)

    peak_no = peak_mem_per_step(no_mem_raw).assign(mode="no_ckpt")
    peak_ck = peak_mem_per_step(ck_mem_raw).assign(mode="ckpt")
    peak_all = pd.concat([peak_no, peak_ck], ignore_index=True)

    # 1) Peak memory per step
    plt.figure()
    for mode, d in peak_all.groupby("mode"):
        d = d.sort_values("global_step")
        plt.plot(d["global_step"], d["mem_mib"], label=mode)
    plt.xlabel("Global step")
    plt.ylabel("Peak memory (MiB)")
    plt.legend()
    plt.title("Peak memory per step: checkpoint vs no-checkpoint")
    plt.savefig(base / "01_peak_memory_per_step.png", bbox_inches="tight")
    plt.close()

    # 2) Step times & epoch averages
    step_no = step_time_from_layer(no_layer_raw).assign(mode="no_ckpt")
    step_ck = step_time_from_layer(ck_layer_raw).assign(mode="ckpt")
    steps_all = pd.concat([step_no, step_ck], ignore_index=True)

    plt.figure()
    for mode, d in steps_all.groupby("mode"):
        d = d.sort_values("global_step")
        plt.plot(d["global_step"], d["ms"], label=mode)
    plt.xlabel("Global step")
    plt.ylabel("Step time (ms)")
    plt.legend()
    plt.title("Step time: checkpoint vs no-checkpoint")
    plt.savefig(base / "02A_step_time_vs_global_step.png", bbox_inches="tight")
    plt.close()

    epoch_time = (steps_all.groupby(["mode","epoch"])["ms"]
                  .mean().reset_index().rename(columns={"ms":"avg_step_ms"}))
    plt.figure()
    for mode, d in epoch_time.groupby("mode"):
        plt.plot(d["epoch"], d["avg_step_ms"], marker="o", label=mode)
    plt.xlabel("Epoch")
    plt.ylabel("Average step time (ms)")
    plt.legend()
    plt.title("Average step time per epoch")
    plt.savefig(base / "02B_avg_step_time_per_epoch.png", bbox_inches="tight")
    plt.close()

    # 3) Extra forward passes
    fwd_no = fwd_counts_per_step(no_layer_raw).assign(mode="no_ckpt")
    fwd_ck = fwd_counts_per_step(ck_layer_raw).assign(mode="ckpt")
    fwd_all = pd.concat([fwd_no, fwd_ck], ignore_index=True)
    avg_fwd = fwd_all.groupby("mode")["fwd_events"].mean().reset_index()

    plt.figure()
    plt.bar(avg_fwd["mode"], avg_fwd["fwd_events"])
    plt.xlabel("Mode")
    plt.ylabel("Avg # forward events per step")
    plt.title("Checkpointing increases forward recomputation")
    plt.savefig(base / "03A_avg_fwd_events_per_step.png", bbox_inches="tight")
    plt.close()

    # Phase sequences for a canonical step (epoch=1, step=1)
    seq_no = phase_sequence_for_step(no_layer_raw, 1, 1)
    seq_ck = phase_sequence_for_step(ck_layer_raw, 1, 1)

    plt.figure()
    plt.plot(seq_no["order"], seq_no["phase_val"])
    plt.yticks([0,1], ["bwd","fwd"])
    plt.xlabel("Operation order within step (epoch=1, step=1)")
    plt.ylabel("Phase")
    plt.title("Phase sequence (no checkpoint)")
    plt.savefig(base / "03B_phase_sequence_no_ckpt.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(seq_ck["order"], seq_ck["phase_val"])
    plt.yticks([0,1], ["bwd","fwd"])
    plt.xlabel("Operation order within step (epoch=1, step=1)")
    plt.ylabel("Phase")
    plt.title("Phase sequence (with checkpoint)")
    plt.savefig(base / "03C_phase_sequence_ckpt.png", bbox_inches="tight")
    plt.close()

    # 4) Epoch concatenation view with boundaries
    plt.figure()
    for mode, d in peak_all.groupby("mode"):
        d = d.sort_values("global_step")
        plt.plot(d["global_step"], d["mem_mib"], label=mode)
    max_steps = max(peak_no["step"].max(), peak_ck["step"].max())
    max_epoch = max(peak_no["epoch"].max(), peak_ck["epoch"].max())
    for e in range(1, max_epoch):
        x = e * max_steps
        plt.axvline(x=x, linestyle="--")
    plt.xlabel("Global step (epochs concatenated)")
    plt.ylabel("Peak memory (MiB)")
    plt.legend()
    plt.title("Epochs concatenated: peak memory cyclicity")
    plt.savefig(base / "04_epochs_concatenated_peak_memory.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize checkpoint vs no-checkpoint training logs.")
    parser.add_argument("--no-ckpt-mem-raw", required=True, dest="no_ckpt_mem_raw")
    parser.add_argument("--no-ckpt-mem-epoch", required=True, dest="no_ckpt_mem_epoch")
    parser.add_argument("--no-ckpt-layer-raw", required=True, dest="no_ckpt_layer_raw")
    parser.add_argument("--no-ckpt-layer-epoch", required=True, dest="no_ckpt_layer_epoch")
    parser.add_argument("--ckpt-mem-raw", required=True, dest="ckpt_mem_raw")
    parser.add_argument("--ckpt-mem-epoch", required=True, dest="ckpt_mem_epoch")
    parser.add_argument("--ckpt-layer-raw", required=True, dest="ckpt_layer_raw")
    parser.add_argument("--ckpt-layer-epoch", required=True, dest="ckpt_layer_epoch")
    parser.add_argument("--outdir", default="figs")
    args = parser.parse_args()
    main(args)
