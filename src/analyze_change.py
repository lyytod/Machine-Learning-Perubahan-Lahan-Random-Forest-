import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from collections import Counter
from config import CLASS_NAMES, CLASS_MAPPING, PIXEL_AREA_M2
from ndvi_to_class import ndvi_to_class


def compute_area_stats(label_array):
    counts = Counter(label_array[label_array != 255].flatten())
    stats = []
    for class_id, count in counts.items():
        class_name = CLASS_NAMES.get(class_id, f"Kelas {class_id}")
        luas_m2 = count * PIXEL_AREA_M2
        luas_ha = luas_m2 / 10000
        stats.append({"Kelas": class_name, "Piksel": count, "Luas (m2)": luas_m2, "Luas (ha)": luas_ha})
    return stats, counts


def compute_transition_matrix(from_array, to_array):
    mask = (from_array != 255) & (to_array != 255)
    flat_from = from_array[mask].flatten()
    flat_to = to_array[mask].flatten()

    labels = sorted(CLASS_NAMES.keys())
    matrix = pd.DataFrame(0, index=[CLASS_NAMES[i] for i in labels], columns=[CLASS_NAMES[i] for i in labels])

    for f, t in zip(flat_from, flat_to):
        matrix.at[CLASS_NAMES[f], CLASS_NAMES[t]] += 1

    return matrix


def save_stats_to_csv(stats, filename):
    df = pd.DataFrame(stats)
    df.to_csv(filename, index=False)


def plot_bar_comparison(counts_from, counts_to, output_dir):
    labels = list(CLASS_NAMES.values())
    luas_from = [v / 10000 for v in counts_from.values()]  # m2 ke ha
    luas_to = [v / 10000 for v in counts_to.values()]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], luas_from, width, label="Tahun Awal")
    plt.bar([i + width/2 for i in x], luas_to, width, label="Tahun Akhir")

    plt.xticks(x, labels)
    plt.ylabel("Luas Tutupan Lahan (ha)")
    plt.title("Perbandingan Tutupan Lahan")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "grafik_perbandingan_luas.png"), dpi=300)
    plt.show()


def plot_pie_chart(stats, tahun, output_dir):
    labels = [s["Kelas"] for s in stats]
    luas_ha = [s["Luas (ha)"] for s in stats]

    plt.figure(figsize=(6, 6))
    plt.pie(luas_ha, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f"Proporsi Tutupan Lahan Tahun {tahun}")
    plt.axis('equal')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"pie_tutupan_lahan_{tahun}.png"), dpi=300)
    plt.show()


def plot_transition_heatmap(csv_path, output_dir):
    df = pd.read_csv(csv_path, index_col=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5)
    plt.title("Matriks Perubahan Kelas")
    plt.xlabel("Tahun Akhir")
    plt.ylabel("Tahun Awal")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "heatmap_perubahan_kelas.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    ndvi_from_path = "data/ndvi_from.tif"
    ndvi_to_path = "data/ndvi_to.tif"
    output_dir = "output"

    with rasterio.open(ndvi_from_path) as src:
        ndvi_from = src.read(1)
    with rasterio.open(ndvi_to_path) as src:
        ndvi_to = src.read(1)

    label_from = ndvi_to_class(ndvi_from)
    label_to = ndvi_to_class(ndvi_to)

    stats_from, counts_from = compute_area_stats(label_from)
    stats_to, counts_to = compute_area_stats(label_to)

    save_stats_to_csv(stats_from, os.path.join(output_dir, "statistik_klasifikasi_from.csv"))
    save_stats_to_csv(stats_to, os.path.join(output_dir, "statistik_klasifikasi_to.csv"))

    transition_matrix = compute_transition_matrix(label_from, label_to)
    transition_matrix.to_csv(os.path.join(output_dir, "matrix_perubahan.csv"))

    plot_bar_comparison(counts_from, counts_to, output_dir)
    plot_pie_chart(stats_from, tahun="awal", output_dir=output_dir)
    plot_pie_chart(stats_to, tahun="akhir", output_dir=output_dir)
    plot_transition_heatmap(os.path.join(output_dir, "matrix_perubahan.csv"), output_dir)
