import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import logging

logger = logging.getLogger(__name__)

def generate_static_map(change_map_path, output_png_path, output_tif_path):
    # Buka raster perubahan lahan
    with rasterio.open(change_map_path) as src:
        perubahan_data = src.read(1)
        meta = src.meta.copy()
        bounds = src.bounds
        # Ambil nilai NoData dari sumber
        nodata_value = src.nodata if src.nodata is not None else 255

    # Mapping warna RGB berdasarkan klasifikasi (dari modul interaktif sebelumnya)
    class_colors_rgb = {
        0: [0, 0, 0],         # 0->0 Non-Vegetasi -> Non-Vegetasi (Hitam)
        1: [255, 127, 14],    # 0->1 Non-Vegetasi -> Vegetasi Sedang (Jingga)
        2: [174, 199, 232],   # 0->2 Non-Vegetasi -> Vegetasi Tinggi (Biru Muda)
        10: [31, 119, 180],    # 1→0 Vegetasi Sedang → Non-Vegetasi (Biru Tua)
        11: [44, 160, 44],     # 1→1 Vegetasi Sedang → Vegetasi Sedang (Hijau)
        12: [148, 103, 189],   # 1→2 Vegetasi Sedang → Vegetasi Tinggi (Ungu)
        20: [227, 119, 194],   # 2→0 Vegetasi Tinggi → Non-Vegetasi (Pink)
        21: [188, 189, 34],    # 2→1 Vegetasi Tinggi → Vegetasi Sedang (Kuning kehijauan)
        22: [158, 218, 229]    # 2→2 Vegetasi Tinggi → Vegetasi Tinggi (Cyan)
    }

    # Buat citra RGB dari data klasifikasi (tanpa downsample di sini, hanya untuk plotting)
    rgb_image = np.zeros((perubahan_data.shape[0], perubahan_data.shape[1], 3), dtype=np.uint8)
    # Pastikan piksel NoData tidak diwarnai atau diwarnai secara transparan jika diinginkan
    # Di sini, kita akan membiarkan piksel NoData tetap 0,0,0 (hitam) seperti default np.zeros
    # Atau, jika nodata_value bukan 0, kita bisa menanganinya secara eksplisit.
    # Untuk saat ini, asumsikan 0,0,0 adalah non-veg non-veg dan nodata diabaikan jika tidak ada di class_colors.
    
    for class_value, color_rgb in class_colors_rgb.items():
        mask = perubahan_data == class_value
        for i in range(3):
            rgb_image[:, :, i][mask] = color_rgb[i]
    
    # Tangani NoData secara eksplisit jika nodata_value tidak sama dengan 0 dan belum ditangani
    if nodata_value is not None and nodata_value != 0 and nodata_value in np.unique(perubahan_data):
        nodata_mask = perubahan_data == nodata_value
        rgb_image[nodata_mask] = [0, 0, 0] # Atur NoData menjadi hitam, atau [255,255,255] putih, atau lainnya

    # --- Buat dan Simpan PNG dengan Legenda --- 
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Tampilkan gambar
    ax.imshow(rgb_image, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    ax.set_title("Peta Perubahan Tutupan Lahan")
    ax.set_xlabel("Bujur")
    ax.set_ylabel("Lintang")
    ax.set_aspect('auto') # Gunakan 'auto' untuk menghindari distorsi jika perbandingan aspek tidak 1:1

    # Buat legenda manual
    legend_elements = []
    # Urutkan berdasarkan kunci untuk konsistensi legenda
    sorted_class_values = sorted(class_colors_rgb.keys())

    legend_labels = {
        0: "Non-Vegetasi → Non-Vegetasi",
        1: "Non-Vegetasi → Vegetasi Sedang",
        2: "Non-Vegetasi → Vegetasi Tinggi",
        10: "Vegetasi Sedang → Non-Vegetasi",
        11: "Vegetasi Sedang → Vegetasi Sedang",
        12: "Vegetasi Sedang → Vegetasi Tinggi",
        20: "Vegetasi Tinggi → Non-Vegetasi",
        21: "Vegetasi Tinggi → Vegetasi Sedang",
        22: "Vegetasi Tinggi → Vegetasi Tinggi"
    }

    for val in sorted_class_values:
        color = np.array(class_colors_rgb[val]) / 255.0 # Normalisasi ke 0-1
        label = legend_labels.get(val, f"Kode {val}")
        legend_elements.append(plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=10, linestyle='None'))

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Kategori Perubahan")
    plt.tight_layout() # Sesuaikan layout agar legenda tidak tumpang tindih

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    try:
        plt.savefig(output_png_path, dpi=300)
        logger.info(f"✅ Peta statis PNG berhasil dibuat: {output_png_path}")
    except OSError as e:
        logger.error(f"[!] Gagal menyimpan peta statis PNG ke {output_png_path}: {e}")
    plt.close(fig)

    # --- Simpan TIF dengan Koordinat yang Sama dengan Sumber --- 
    meta.update({
        'dtype': perubahan_data.dtype,  # Pastikan dtype sesuai dengan data asli
        'count': 1,  # TIF hanya memiliki 1 band (data perubahan)
        'nodata': nodata_value, # Pertahankan nilai nodata dari sumber
        # transform, crs, width, height sudah ada di meta dari src.meta.copy()
    })

    os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)
    try:
        with rasterio.open(output_tif_path, 'w', **meta) as dst:
            dst.write(perubahan_data, 1)
        logger.info(f"✅ Peta perubahan TIF berhasil dibuat: {output_tif_path}")
    except OSError as e:
        logger.error(f"[!] Gagal menyimpan peta perubahan TIF ke {output_tif_path}: {e}")

if __name__ == "__main__":
    # Contoh penggunaan (pastikan ada data/change_map.tif)
    # from src.utils import setup_logger
    # setup_logger("outputs/static_map_log.txt")
    # generate_static_map(
    #     change_map_path="data/change_map.tif", 
    #     output_png_path="outputs/static_change_map.png", 
    #     output_tif_path="outputs/static_change_map.tif"
    # )
    logger.info("Ini adalah modul untuk membuat peta statis. Biasanya dipanggil dari main.py.") 