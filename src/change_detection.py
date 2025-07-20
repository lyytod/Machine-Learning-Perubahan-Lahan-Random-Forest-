import rasterio
import numpy as np
import pandas as pd
import os

def detect_change(label_from_path, label_to_path, output_raster_path, stats_csv_path):
    with rasterio.open(label_from_path) as src_from, rasterio.open(label_to_path) as src_to:
        label_from = src_from.read(1)
        label_to = src_to.read(1)
        profile = src_from.profile

    if label_from.shape != label_to.shape:
        raise ValueError("Ukuran label_from dan label_to tidak sama")

    # Hitung perubahan (kode: from * 10 + to)
    change_map = (label_from.astype(np.int16) * 10) + label_to.astype(np.int16)

    # Update profil raster output
    profile.update(dtype=rasterio.int16, count=1)

    os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
    with rasterio.open(output_raster_path, "w", **profile) as dst:
        dst.write(change_map, 1)

    # Hitung statistik perubahan
    unique, counts = np.unique(change_map, return_counts=True)
    stats = []

    label_dict = {
        0: "Non-Vegetasi",
        1: "Vegetasi Sedang",
        2: "Vegetasi Tinggi"
    }

    pixel_area = abs(profile["transform"][0] * profile["transform"][4])  # misal: 100 m² untuk resolusi 10x10

    for code, count in zip(unique, counts):
        dari = code // 10
        ke = code % 10
        stats.append({
            "Kode Perubahan": f"{dari} → {ke}",
            "Dari": label_dict.get(dari, f"Kode {dari}"),
            "Ke": label_dict.get(ke, f"Kode {ke}"),
            "Jumlah Pixel": count,
            "Luas (m²)": count * pixel_area,
            "Luas (ha)": count * pixel_area / 10_000
        })

    df = pd.DataFrame(stats)
    df.to_csv(stats_csv_path, index=False)
    print(f"✅ Deteksi perubahan selesai. Hasil disimpan ke: {output_raster_path} dan {stats_csv_path}")
