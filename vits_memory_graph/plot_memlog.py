import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

EPOCH_AVG_CSV = Path("./mem_logs/memlog_epoch_avg.csv")
PLOT_PNG = Path("./mem_logs/memplot_epochs_concat.png")

def main():
    if not EPOCH_AVG_CSV.exists():
        raise FileNotFoundError(f"{EPOCH_AVG_CSV} not found. Сначала запустите обучение.")

    # читаем CSV (epoch, x_label, phase, layer, mem_mib)
    data = defaultdict(list)  # epoch -> list[(x_label, mem)]
    with open(EPOCH_AVG_CSV, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ep = int(row["epoch"])
            xl = row["x_label"]
            mem = float(row["mem_mib"])
            data[ep].append((xl, mem))

    epochs = sorted(data.keys())
    first_epoch = epochs[0]

    # порядок слоёв: fwd-L1..Ln, bwd-Ln..L1 — строим по первой эпохе
    def layer_key(xl: str):
        if xl.startswith("fwd-"):
            return (0, int(xl.split("-L")[-1]))
        return (1, -int(xl.split("-L")[-1]))

    order = [xl for xl, _ in sorted(data[first_epoch], key=lambda t: layer_key(t[0]))]
    points_per_epoch = len(order)
    gap = 2  # небольшой зазор между эпохами (можно 0)

    # склеиваем все эпохи в один x-ряд
    xs, ys, labels = [], [], []
    for e_idx, ep in enumerate(epochs):
        offset = e_idx * (points_per_epoch + gap)
        series = {xl: mem for xl, mem in data[ep]}
        for j, xl in enumerate(order):
            xs.append(offset + j)
            ys.append(series[xl])
            labels.append(f"e{ep}:{xl}")

    plt.figure(figsize=(18, 8))
    plt.plot(xs, ys, linewidth=2)

    # вертикальные линии на границах эпох
    for e_idx in range(1, len(epochs)):
        x = e_idx * (points_per_epoch + gap) - (gap / 2)
        plt.axvline(x=x, linestyle="--", alpha=0.3)

    # тики пореже, чтобы не захламлять ось
    stride = max(1, len(labels) // 60)  # нацелимся ~60 тиков
    tick_pos = list(range(0, len(labels), stride))
    plt.xticks(tick_pos, [labels[i] for i in tick_pos], rotation=60, ha="right")

    plt.ylabel("GPU memory, MiB")
    plt.xlabel("Layer timeline concatenated across epochs")
    plt.title("ViT-S/16 on CIFAR-100: GPU memory per block (avg over steps) — epochs concatenated")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=150)
    print(f"Saved: {PLOT_PNG}")

if __name__ == "__main__":
    main()
