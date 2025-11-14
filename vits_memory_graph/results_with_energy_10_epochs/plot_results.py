#!/usr/bin/env python
"""
Сравнение метрик ViT_S в трёх режимах:
- Без чекпоинта
- С чекпоинтом
- С чекпоинтом и лимитом 2 ГБ

Ожидаемые файлы (по умолчанию в той же папке, где запускается скрипт):
    - no_check_epoch_metrics.csv
    - check_epoch_metrics.csv
    - 2gb_check_epoch_metrics.csv

Пример запуска:
    python compare_epoch_metrics.py \
        --data-dir . \
        --plots-dir ./plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODES = {
    "no_check": "no_check_epoch_metrics.csv",
    "check": "check_epoch_metrics.csv",
    "check_2gb": "2gb_check_epoch_metrics.csv",
}


def load_data(data_dir: Path):
    """Загружаем все три CSV в словарь {mode: DataFrame}."""
    dfs = {}
    for mode, fname in MODES.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        df = pd.read_csv(path)
        dfs[mode] = df.sort_values("epoch").reset_index(drop=True)
    return dfs


def get_common_metrics(dfs):
    """Находим пересечение колонок (метрик) во всех режимах."""
    cols_sets = [set(df.columns) for df in dfs.values()]
    common = set.intersection(*cols_sets)
    # epoch — служебный столбец, остальные считаем метриками
    metrics = sorted(c for c in common if c != "epoch")
    return metrics


def print_epochwise_summary(dfs, metrics, baseline_mode="no_check"):
    """Печатаем сравнение по каждой метрике (по последней эпохе и по среднему)."""
    print("=" * 80)
    print("СРАВНЕНИЕ МЕТРИК ПО РЕЖИМАМ (последняя эпоха и среднее по эпохам)")
    print("=" * 80)
    print(f"Режимы: {', '.join(dfs.keys())}")
    print(f"Базовый режим для относительных величин: {baseline_mode}")
    print()

    # Проверим, что у всех одинаковый набор эпох (на всякий случай)
    epochs_sets = {mode: set(df["epoch"]) for mode, df in dfs.items()}
    if len({tuple(sorted(s)) for s in epochs_sets.values()}) != 1:
        print("ВНИМАНИЕ: наборы эпох в CSV не совпадают между режимами!")
    epochs = sorted(next(iter(epochs_sets.values())))

    for metric in metrics:
        print("-" * 80)
        print(f"МЕТРИКА: {metric}")
        print("Последняя эпоха:")

        # значения метрики в последней эпохе
        last_values = {}
        for mode, df in dfs.items():
            last_row = df.sort_values("epoch").iloc[-1]
            val = float(last_row[metric])
            last_values[mode] = val

        # средние по эпохам
        mean_values = {mode: float(df[metric].mean()) for mode, df in dfs.items()}

        base_last = last_values.get(baseline_mode, None)
        base_mean = mean_values.get(baseline_mode, None)

        # Печатаем табличку
        print(f"{'Mode':<12} {'last_epoch':>15} {'mean':>15} {'last_rel':>12} {'mean_rel':>12}")
        for mode in dfs.keys():
            lv = last_values[mode]
            mv = mean_values[mode]
            if base_last and base_last != 0:
                last_rel = lv / base_last
            else:
                last_rel = float("nan")
            if base_mean and base_mean != 0:
                mean_rel = mv / base_mean
            else:
                mean_rel = float("nan")

            print(
                f"{mode:<12} "
                f"{lv:15.4f} "
                f"{mv:15.4f} "
                f"{last_rel:12.3f} "
                f"{mean_rel:12.3f}"
            )

        print()

    # Дополнительно — суммарное время и энергия за все эпохи
    print("=" * 80)
    print("ИНТЕГРАЛЬНЫЕ ПОКАЗАТЕЛИ (сумма по всем эпохам)")
    print("=" * 80)

    integral_metrics = [
        "total_time_s",
        "train_time_s",
        "eval_time_s",
        "total_energy_j",
        "train_energy_j",
        "eval_energy_j",
    ]
    integral_metrics = [m for m in integral_metrics if m in metrics]

    for metric in integral_metrics:
        print("-" * 80)
        print(f"ИНТЕГРАЛЬНАЯ МЕТРИКА: {metric}")
        sums = {mode: float(df[metric].sum()) for mode, df in dfs.items()}
        base_sum = sums.get(baseline_mode, None)

        print(f"{'Mode':<12} {'sum':>15} {'rel_to_base':>15}")
        for mode in dfs.keys():
            s = sums[mode]
            if base_sum and base_sum != 0:
                rel = s / base_sum
            else:
                rel = float("nan")
            print(f"{mode:<12} {s:15.4f} {rel:15.3f}")
        print()


def plot_metrics(dfs, metrics, plots_dir: Path):
    """Рисуем графики каждой метрики по эпохам для трёх режимов."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        fig, ax = plt.subplots()
        for mode, df in dfs.items():
            ax.plot(
                df["epoch"],
                df[metric],
                marker="o",
                label=mode,
            )

        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)

        out_path = plots_dir / f"{metric}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Сравнение метрик ViT_S для разных режимов чекпоинтинга."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Папка, где лежат CSV файлы *_epoch_metrics.csv",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Куда сохранить графики (если не указано — графики не строятся)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="no_check",
        choices=list(MODES.keys()),
        help="Базовый режим для относительных сравнений",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    dfs = load_data(data_dir)
    metrics = get_common_metrics(dfs)

    print_epochwise_summary(dfs, metrics, baseline_mode=args.baseline)

    if args.plots_dir is not None:
        plots_dir = Path(args.plots_dir)
        print(f"Рисую графики в: {plots_dir}")
        plot_metrics(dfs, metrics, plots_dir)
        print("Графики сохранены.")


if __name__ == "__main__":
    main()
