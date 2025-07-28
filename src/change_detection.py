import rasterio
import numpy as np
import pandas as pd
import os
from src.config import CLASS_NAMES

def detect_change(label_from_path, label_to_path, output_raster_path, stats_csv_path):
    with rasterio.open(label_from_path) as src_from, rasterio.open(label_to_path) as src_to:
        label_from = src_from.read(1)
        label_to = src_to.read(1)
        profile = src_from.profile
        nodata_value = src_from.nodata if src_from.nodata is not None else 255 # Assume 255 if not explicitly set

    if label_from.shape != label_to.shape:
        raise ValueError("Ukuran label_from dan label_to tidak sama")

    # Inisialisasi peta perubahan dengan nilai nodata
    change_map = np.full(label_from.shape, nodata_value, dtype=np.int16)

    # Buat mask untuk piksel yang valid (bukan nodata di kedua label)
    valid_mask = (label_from != nodata_value) & (label_to != nodata_value)

    # Hitung perubahan hanya untuk piksel yang valid
    change_map[valid_mask] = (label_from[valid_mask].astype(np.int16) * 10) + label_to[valid_mask].astype(np.int16)

    # Update profil raster output
    profile.update(dtype=rasterio.int16, count=1, nodata=nodata_value)

    os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
    with rasterio.open(output_raster_path, "w", **profile) as dst:
        dst.write(change_map, 1)

    # Hitung statistik perubahan
    # Filter nodata dari perhitungan statistik
    unique, counts = np.unique(change_map[change_map != nodata_value], return_counts=True)
    stats = []

    # Gunakan CLASS_NAMES dari config
    # label_dict = {
    #     0: "Non-Vegetasi",
    #     1: "Vegetasi Sedang",
    #     2: "Vegetasi Tinggi"
    # }

    pixel_area = abs(profile["transform"][0] * profile["transform"][4])

    for code, count in zip(unique, counts):
        dari = code // 10
        ke = code % 10
        stats.append({
            "Kode Perubahan": f"{dari} → {ke}",
            "Dari": CLASS_NAMES.get(dari, f"Kode {dari}"),
            "Ke": CLASS_NAMES.get(ke, f"Kode {ke}"),
            "Jumlah Pixel": count,
            "Luas (m²)": count * pixel_area,
            "Luas (ha)": count * pixel_area / 10_000
        })

    df = pd.DataFrame(stats)
    df.to_csv(stats_csv_path, index=False)
    print(f"✅ Deteksi perubahan selesai. Hasil disimpan ke: {output_raster_path} dan {stats_csv_path}")
