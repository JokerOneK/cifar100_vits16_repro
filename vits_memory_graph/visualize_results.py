import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob


def main():
    log_dir = "./mem_logs"
    # Находим все файлы, начинающиеся с metrics_
    files = glob.glob(f"{log_dir}/*.csv")

    if not files:
        print("CSV файлы не найдены в", log_dir)
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Извлекаем имя метода из имени файла если столбец method пустой или для уверенности
            method_name = Path(f).stem.replace("metrics_", "")
            if 'method' not in df.columns:
                df['method'] = method_name
            dfs.append(df)
        except Exception as e:
            print(f"Ошибка чтения {f}: {e}")

    if not dfs:
        return

    full_df = pd.concat(dfs, ignore_index=True)

    # Настройка стиля
    sns.set_theme(style="whitegrid")

    # Создаем холст: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Сравнение методов PEFT (ViT-Small)', fontsize=16)

    # 1. Accuracy Curve (Рост точности по эпохам)
    sns.lineplot(data=full_df, x='epoch', y='test_acc_pct', hue='method', marker='o', ax=axes[0, 0], linewidth=2.5)
    axes[0, 0].set_title("Test Accuracy per Epoch")
    axes[0, 0].set_ylabel("Accuracy (%)")

    # 2. Average Train Time (Bar Chart)
    # Считаем среднее время на эпоху для каждого метода
    avg_metrics = full_df.groupby('method')[
        ['train_time_s', 'train_energy_j', 'avg_power_w', 'test_acc_pct']].mean().reset_index()

    sns.barplot(data=avg_metrics, x='method', y='train_time_s', ax=axes[0, 1], palette="viridis")
    axes[0, 1].set_title("Average Training Time per Epoch (sec)")
    axes[0, 1].bar_label(axes[0, 1].containers[0], fmt='%.1f')

    # 3. Average Energy (Bar Chart)
    sns.barplot(data=avg_metrics, x='method', y='train_energy_j', ax=axes[1, 0], palette="magma")
    axes[1, 0].set_title("Average Energy Consumption per Epoch (Joules)")
    axes[1, 0].bar_label(axes[1, 0].containers[0], fmt='%.0f')

    # 4. Final Accuracy vs Efficiency (Scatter Plot)
    # Ось X - Энергия, Ось Y - Точность. Чем выше и левее - тем лучше.
    sns.scatterplot(data=avg_metrics, x='train_energy_j', y='test_acc_pct', hue='method', s=200, style='method',
                    ax=axes[1, 1])
    axes[1, 1].set_title("Efficiency Frontier: Accuracy vs Energy")
    axes[1, 1].set_xlabel("Energy (J) per Epoch")
    axes[1, 1].set_ylabel("Avg Accuracy (%)")

    plt.tight_layout()
    output_path = "peft_comparison_results.png"
    plt.savefig(output_path)
    print(f"Графики сохранены в {output_path}")
    plt.show()


if __name__ == "__main__":
    main()