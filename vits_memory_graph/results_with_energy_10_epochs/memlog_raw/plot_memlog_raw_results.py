#!/usr/bin/env python
"""
Анализ потребления GPU памяти по слоям для 3 режимов:
- без чекпоинта
- с чекпоинтом
- с чекпоинтом и лимитом 2 ГБ

Ожидаемые файлы в data_dir:
    - no_check_memlog_raw.csv.gz
    - chech_memlog_raw.csv.gz
    - 2_gb_check_memlog_raw.csv.gz

Пример запуска:
    python plot_layer_mem_patterns.py --data-dir . --out-dir ./plots_layer
"""

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------

def read_memlog(path: Path) -> pd.DataFrame:
    """Чтение сжатого CSV с колонками: epoch, step, phase, layer, mem_mib."""
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f)
    return df


def add_phase_ext_per_step(df: pd.DataFrame, has_checkpoint: bool) -> pd.DataFrame:
    """
    Для каждого (epoch, step) помечаем phase_ext:
      - без чекпоинта:
            fwd_main / bwd
      - с чекпоинтом:
            до первого bwd: fwd_main
            после первого bwd:
                phase=fwd  -> fwd_recompute (forward во время backward)
                phase=bwd  -> bwd
    """

    def _process_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_index()  # сохраняем порядок как в логе
        if not has_checkpoint:
            g = g.copy()
            g["phase_ext"] = np.where(g["phase"] == "fwd", "fwd_main", "bwd")
            return g

        seen_bwd = False
        phase_ext = []
        for _, row in g.iterrows():
            if not seen_bwd:
                if row["phase"] == "bwd":
                    seen_bwd = True
                    phase_ext.append("bwd")
                else:
                    phase_ext.append("fwd_main")
            else:
                if row["phase"] == "fwd":
                    phase_ext.append("fwd_recompute")
                else:
                    phase_ext.append("bwd")

        g = g.copy()
        g["phase_ext"] = phase_ext
        return g

    return df.groupby(["epoch", "step"], group_keys=False).apply(_process_group)


def compute_mean_step_pattern(df_ext: pd.DataFrame):
    """
    Средний паттерн одного training step:
      группируем по (phase_ext, layer), считаем средний mem_mib.

    Возвращаем:
      - agg: DataFrame с колонками [phase_ext, layer, mem_mib, pos]
      - order: список (phase_ext, layer) в порядке следования операций.
    """
    # средний mem_mib по всем эпохам и шагам
    agg = (
        df_ext
        .groupby(["phase_ext", "layer"], as_index=False)["mem_mib"]
        .mean()
    )

    max_layer = int(agg["layer"].max())

    # строим логический порядок операций:
    #   F1..F_L  -> F_recompute(L..1) -> B(L..1)
    order = []

    # основной forward: 1..L
    for l in range(1, max_layer + 1):
        if ((agg["phase_ext"] == "fwd_main") & (agg["layer"] == l)).any():
            order.append(("fwd_main", l))

    # recompute forward: L..1 (только если он есть в данном режиме)
    for l in range(max_layer, 0, -1):
        if ((agg["phase_ext"] == "fwd_recompute") & (agg["layer"] == l)).any():
            order.append(("fwd_recompute", l))

    # backward: L..1
    for l in range(max_layer, 0, -1):
        if ((agg["phase_ext"] == "bwd") & (agg["layer"] == l)).any():
            order.append(("bwd", l))

    # назначаем индекс позиции
    pos_map = {(ph, l): i for i, (ph, l) in enumerate(order)}

    agg["pos"] = agg.apply(
        lambda r: pos_map.get((r["phase_ext"], int(r["layer"])), np.nan),
        axis=1,
    )
    agg = agg.dropna(subset=["pos"]).sort_values("pos").reset_index(drop=True)

    return agg, order


def make_xtick_labels(order):
    """
    Из последовательности (phase_ext, layer) делаем подписи оси X:
      fwd_main      -> F{layer}
      fwd_recompute -> R{layer}
      bwd           -> B{layer}
    """
    labels = []
    for phase_ext, layer in order:
        if phase_ext == "fwd_main":
            prefix = "F"
        elif phase_ext == "fwd_recompute":
            prefix = "R"  # R = recompute
        else:
            prefix = "B"
        labels.append(f"{prefix}{layer}")
    return labels


# ------------------------------------------------------------
# Построение графиков
# ------------------------------------------------------------

def plot_per_mode(agg: pd.DataFrame, order, mode_name: str, out_dir: Path):
    """График для одного режима: F -> R -> B, по слоям."""
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = make_xtick_labels(order)
    x = agg["pos"].values
    y = agg["mem_mib"].values

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, marker="o", linewidth=1.5, label=mode_name)

    # линия в 2 ГБ
    ax.axhline(2048.0, linestyle="--", color="red", linewidth=1.2, label="2 GiB limit")

    ax.set_title(f"Average memory pattern of one training step ({mode_name})")
    ax.set_ylabel("GPU memory, MiB")
    ax.set_xlabel("Operations within one step (layer/pahse)")

    # аккуратно ставим подписи по X
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_dir / f"mem_pattern_{mode_name}.png", dpi=150)
    plt.close(fig)


def plot_overlay(patterns, out_dir: Path):
    """
    Общий график сравнения:
      - ось X нормируем в [0, 1] для каждого режима (чтобы «растянуть» без чекпоинта)
      - ось Y — mem_mib
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    for mode_name, (agg, order) in patterns.items():
        x = agg["pos"].values
        y = agg["mem_mib"].values

        # нормируем X в 0..1, чтобы все кривые заканчивались в одной точке
        x_norm = x / x.max()
        ax.plot(
            x_norm,
            y,
            marker="o",
            linewidth=1.5,
            label=mode_name,
        )

    # линия в 2 ГБ
    ax.axhline(2048.0, linestyle="--", color="red", linewidth=1.2, label="2 GiB limit")

    ax.set_title("One step memory pattern comparison (normalized time)")
    ax.set_xlabel("Normalized time within 1 training step (0..1)")
    ax.set_ylabel("GPU memory, MiB")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_dir / "mem_pattern_overlay_normalized.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Сравнение паттернов памяти по слоям для трёх режимов."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Папка с файлами *_memlog_raw.csv.gz",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./plots_layer",
        help="Куда сохранить графики и агрегированные CSV",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    # пути к файлам
    path_no = data_dir / "no_check_memlog_raw.csv.gz"
    path_chk = data_dir / "chech_memlog_raw.csv.gz"
    path_chk2 = data_dir / "2_gb_check_memlog_raw.csv.gz"

    # --- загружаем и помечаем расширенную фазу ---
    df_no = read_memlog(path_no)
    df_chk = read_memlog(path_chk)
    df_chk2 = read_memlog(path_chk2)

    df_no_ext = add_phase_ext_per_step(df_no, has_checkpoint=False)
    df_chk_ext = add_phase_ext_per_step(df_chk, has_checkpoint=True)
    df_chk2_ext = add_phase_ext_per_step(df_chk2, has_checkpoint=True)

    # --- считаем средний паттерн одного шага ---
    patterns = {}
    for mode_name, df_ext in [
        ("no_check", df_no_ext),
        ("check", df_chk_ext),
        ("check_2gb", df_chk2_ext),
    ]:
        agg, order = compute_mean_step_pattern(df_ext)
        patterns[mode_name] = (agg, order)

        # сохраняем csv для дебага / отчёта
        out_dir.mkdir(parents=True, exist_ok=True)
        agg.to_csv(out_dir / f"avg_step_pattern_{mode_name}.csv", index=False)

        # рисуем отдельный график
        plot_per_mode(agg, order, mode_name, out_dir)

    # общий график-сравнение (с растяжкой по X)
    plot_overlay(patterns, out_dir)


if __name__ == "__main__":
    main()
