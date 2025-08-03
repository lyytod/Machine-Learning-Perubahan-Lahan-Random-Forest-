import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from collections import Counter
from src.config import CLASS_NAMES, CLASS_MAPPING, CLASS_COLORS
from src.ndvi_to_class import ndvi_to_class

def compute_area_stats(label_array, pixel_area):
    counts = Counter(label_array[label_array != 255].flatten())
    stats = []
    # Pastikan urutan kelas konsisten dengan CLASS_NAMES/CLASS_COLORS
    # Kita akan mengurutkan hasil berdasarkan ID kelas untuk konsistensi
    sorted_class_ids = sorted(CLASS_NAMES.keys())
    
    for class_id in sorted_class_ids:
        class_name = CLASS_NAMES.get(class_id, f"Kelas {class_id}")
        count = counts.get(class_id, 0) # Ambil count, jika tidak ada, default ke 0
        luas_m2 = count * pixel_area
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


def plot_bar_comparison(stats_from, stats_to, output_dir):
    labels = [s["Kelas"] for s in stats_from] # Asumsi kelas sama
    luas_from = [s["Luas (ha)"] for s in stats_from]
    luas_to = [s["Luas (ha)"] for s in stats_to]

    # Dapatkan warna yang konsisten
    colors = [CLASS_COLORS[label] for label in labels]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], luas_from, width, label="Tahun Awal", color=[c for c in colors]) # Aplikasikan warna
    plt.bar([i + width/2 for i in x], luas_to, width, label="Tahun Akhir", color=[c for c in colors])   # Aplikasikan warna

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
    # Urutan kelas yang diinginkan untuk plotting (sesuai dengan gambar contoh)
    desired_plot_order = ["Vegetasi Tinggi", "Non-Vegetasi", "Vegetasi Sedang"]

    # Buat kamus untuk akses cepat berdasarkan nama kelas
    stats_dict = {s["Kelas"]: s["Luas (ha)"] for s in stats}

    # Susun ulang labels, luas_ha, dan colors sesuai urutan yang diinginkan
    ordered_labels = []
    ordered_luas_ha = []
    ordered_colors = []

    for class_name in desired_plot_order:
        if class_name in stats_dict:
            ordered_labels.append(class_name)
            ordered_luas_ha.append(stats_dict[class_name])
            ordered_colors.append(CLASS_COLORS[class_name])
        # Optional: handle classes that might not be present in stats_dict if necessary

    plt.figure(figsize=(8, 8)) # Ukuran figur yang sedikit lebih besar untuk visualisasi yang lebih baik
    plt.pie(ordered_luas_ha, labels=ordered_labels, autopct='%1.1f%%', startangle=90, colors=ordered_colors, counterclock=False)
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

    # Dapatkan pixel_area dari raster asli untuk perhitungan statistik
    with rasterio.open(ndvi_from_path) as src_meta:
        pixel_area = abs(src_meta.transform[0] * src_meta.transform[4]) # Hitung area piksel

    stats_from, counts_from = compute_area_stats(label_from, pixel_area)
    stats_to, counts_to = compute_area_stats(label_to, pixel_area)

    save_stats_to_csv(stats_from, os.path.join(output_dir, "statistik_klasifikasi_from.csv"))
    save_stats_to_csv(stats_to, os.path.join(output_dir, "statistik_klasifikasi_to.csv"))

    transition_matrix = compute_transition_matrix(label_from, label_to)
    transition_matrix.to_csv(os.path.join(output_dir, "matrix_perubahan.csv"))

    plot_bar_comparison(stats_from, stats_to, output_dir)
    plot_pie_chart(stats_from, tahun="awal", output_dir=output_dir)
    plot_pie_chart(stats_to, tahun="akhir", output_dir=output_dir)
    plot_transition_heatmap(os.path.join(output_dir, "matrix_perubahan.csv"), output_dir)
