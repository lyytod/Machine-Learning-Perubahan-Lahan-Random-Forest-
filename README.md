# Deteksi Perubahan Tutupan Lahan Menggunakan Machine Learning

Proyek ini bertujuan untuk mendeteksi dan menganalisis perubahan tutupan lahan menggunakan citra satelit (NDVI dan RGB) dan teknik Machine Learning, serta menghasilkan visualisasi dari perubahan tersebut.

## Fitur Utama

*   **Dua Mode Operasi:** Pilih antara menjalankan seluruh pipeline (termasuk pelatihan model baru) atau menggunakan model yang sudah ada untuk prediksi di area baru.
*   **Preprocessing Citra:** Pemotongan (clipping) citra satelit (NDVI dan RGB) berdasarkan batas area studi. Termasuk deteksi otomatis dan reprojeksi sistem koordinat ke UTM yang sesuai jika data input belum dalam proyeksi UTM.
*   **Klasifikasi NDVI:** Mengklasifikasikan nilai NDVI menjadi kategori tutupan lahan (Non-Vegetasi, Vegetasi Sedang, Vegetasi Tinggi).
*   **Deteksi Perubahan:** Mengidentifikasi dan menguantifikasi perubahan tutupan lahan antara dua periode waktu.
*   **Pelatihan & Prediksi Model:** Melatih model klasifikasi (Random Forest) menggunakan citra RGB dan label NDVI, kemudian menggunakannya untuk memprediksi tutupan lahan. Mendukung prediksi di area geografis yang berbeda menggunakan model yang sama.
*   **Evaluasi Model:** Mengevaluasi kinerja model klasifikasi.
*   **Analisis Statistik:** Menghitung statistik area dan membuat matriks transisi perubahan tutupan lahan.
*   **Visualisasi:** Menghasilkan grafik statistik dan peta perubahan statis (PNG dan GeoTIFF).

## Struktur Proyek

```
.
├── config.yaml
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── ndvi_from.tif
│   ├── ndvi_to.tif
│   ├── rgb.tif
│   ├── boundary.shp
│   ├── rgb_from_new_area.tif # Opsional, untuk Mode 2
│   ├── rgb_to_new_area.tif   # Opsional, untuk Mode 2
│   └── boundary_new_area.shp # Opsional, untuk Mode 2
├── output/ (folder ini akan dihasilkan setelah menjalankan script)
│   ├── preprocessed/
│   ├── classified/
│   ├── model/
│   ├── prediction/
│   │   └── new_area_prediction_from.tif # Output Mode 2
│   │   └── new_area_prediction_to.tif   # Output Mode 2
│   ├── evaluation/
│   ├── analysis/
│   │   └── new_area_analysis/ # Output Mode 2
│   ├── visualization/
│   │   └── new_area_visualization/ # Output Mode 2
│   ├── change_map_new_area.tif # Output Mode 2
│   └── log.txt
└── src/
    ├── analyze_change.py
    ├── change_detection.py
    ├── config.py
    ├── evaluate.py
    ├── generate_static_map.py
    ├── model.py
    ├── ndvi_to_class.py
    ├── predict.py
    ├── preprocessing.py
    ├── utils.py
    └── visualize_map.py
```

*   `config.yaml`: Berisi jalur input data dan jalur output untuk semua hasil yang dihasilkan, termasuk konfigurasi untuk mode area baru.
*   `main.py`: Skrip utama yang mengorkestrasi seluruh alur kerja deteksi perubahan, mendukung dua mode operasi.
*   `requirements.txt`: Daftar pustaka Python yang dibutuhkan.
*   `data/`: Direktori untuk menyimpan data input mentah (citra NDVI, RGB, dan shapefile batas). Juga dapat berisi data untuk area baru.
*   `output/`: Direktori tempat semua hasil pemrosesan, model, prediksi, analisis, dan visualisasi akan disimpan.
*   `src/`: Berisi modul-modul Python terpisah yang mengimplementasikan setiap langkah dalam alur kerja:
    *   `preprocessing.py`: Fungsi untuk memproses awal citra, termasuk reprojeksi sistem koordinat otomatis ke UTM dan pemotongan citra.
    *   `ndvi_to_class.py`: Klasifikasi nilai NDVI ke kelas tutupan lahan.
    *   `change_detection.py`: Deteksi perubahan antara dua peta klasifikasi.
    *   `model.py`: Ekstraksi fitur, pelatihan, dan penyimpanan model Machine Learning.
    *   `predict.py`: Melakukan prediksi tutupan lahan menggunakan model yang sudah dilatih.
    *   `evaluate.py`: Evaluasi kinerja model.
    *   `analyze_change.py`: Analisis statistik perubahan tutupan lahan.
    *   `visualization.py`: Generasi grafik statistik.
    *   `generate_static_map.py`: Pembuatan peta perubahan statis (menggantikan peta interaktif).
    *   `utils.py`: Fungsi-fungsi utilitas umum.
    *   `config.py`: Definisi nama kelas dan konstanta lainnya.

## Instalasi

Untuk menjalankan proyek ini, Anda harus memiliki Python 3.x terinstal di perangkat Anda. Setelah itu, ikuti langkah-langkah berikut:

1.  **Clone repositori:**
    ```bash
    git clone https://github.com/lyytod/Machine-Learning-Perubahan-Lahan-Random-Forest-.git
    cd land_cover_template
    ```

2.  **Buat dan aktifkan virtual environment (direkomendasikan):**
    ```bash
    python -m venv venv
    # Di Windows
    .\venv\Scripts\activate
    # Di macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

## Penggunaan

1.  **Siapkan Data Input:**
    Tempatkan citra NDVI (`ndvi_from.tif`, `ndvi_to.tif`), citra RGB (`rgb.tif`), dan shapefile batas area studi (`boundary.shp`) di dalam folder `data/` sesuai dengan jalur yang ditentukan dalam `config.yaml`.
    
    Untuk **Mode 2 (Prediksi Area Baru)**, siapkan juga citra RGB periode awal (`rgb_from_new_area.tif`), citra RGB periode akhir (`rgb_to_new_area.tif`), dan shapefile batas area baru (`boundary_new_area.shp`) di folder `data/`.

2.  **Jalankan Skrip Utama:**
    ```bash
    python main.py
    ```

    Anda akan diminta untuk memilih mode operasi:
    *   **1. Jalankan Seluruh Pipeline (Termasuk Pelatihan Model Baru):** Akan menjalankan semua langkah dari preprocessing hingga pembuatan peta statis, termasuk melatih model baru. Ini membutuhkan data `ndvi_from.tif`, `ndvi_to.tif`, `rgb.tif`, dan `boundary.shp`.
    *   **2. Prediksi Menggunakan Model yang Sudah Ada (untuk area lain):** Akan menggunakan model yang sudah ada (dari Mode 1) untuk memprediksi tutupan lahan di area baru dan kemudian menganalisis perubahannya. Ini membutuhkan `rgb_from_new_area.tif`, `rgb_to_new_area.tif`, `boundary_new_area.shp`, dan model yang sudah dilatih dari Mode 1.

## Konfigurasi

File `config.yaml` berisi semua jalur file input dan output. Pastikan untuk memperbarui jalur ini jika struktur folder Anda berbeda atau jika Anda menggunakan nama file yang berbeda.

```yaml
# Contoh isi config.yaml
paths:
  ndvi_from: data/ndvi_from.tif
  ndvi_to: data/ndvi_to.tif
  rgb: data/rgb.tif
  boundary: data/boundary.shp
  change_map: output/change_map.tif

  # Paths for new area prediction (Mode 2)
  rgb_from_new_area: data/rgb_from_new_area.tif
  rgb_to_new_area: data/rgb_to_new_area.tif
  boundary_new_area: data/boundary_new_area.shp

outputs:
  preprocessed: output/preprocessed
  classified: output/classified
  model: output/model
  prediction: output/prediction/prediction.tif
  evaluation: output/evaluation
  analysis: output/analysis
  visualization: output/visualization
  # interactive_map: output/peta_perubahan_interaktif.html # Dihapus, diganti peta statis

  # Outputs for new area prediction (Mode 2)
  prediction_new_area_from: output/prediction/new_area_prediction_from.tif
  prediction_new_area_to: output/prediction/new_area_prediction_to.tif
  change_map_new_area: output/change_map_new_area.tif
  analysis_new_area: output/analysis/new_area_analysis
  visualization_new_area: output/visualization/new_area_visualization
```

## Output

Setelah eksekusi berhasil, folder `output/` akan berisi:

*   `output/preprocessed/`: Citra NDVI dan RGB yang telah dipotong dan direprojeksi.
*   `output/classified/`: Citra NDVI yang telah diklasifikasikan ke dalam kelas tutupan lahan.
*   `output/model/`: Model Machine Learning yang telah dilatih (`random_forest.pkl`).
*   `output/prediction/`: Peta prediksi tutupan lahan (`prediction.tif`). Untuk Mode 2, akan ada `new_area_prediction_from.tif` dan `new_area_prediction_to.tif`.
*   `output/evaluation/`: Laporan evaluasi model.
*   `output/analysis/`: File CSV berisi statistik luas perubahan (`luas_perubahan.csv`) dan matriks transisi (`matrix_perubahan.csv`), serta statistik klasifikasi (`statistik_klasifikasi_from.csv`, `statistik_klasifikasi_to.csv`). Untuk Mode 2, output akan berada di `output/analysis/new_area_analysis/`.
    **Catatan Penting untuk File CSV Analisis:** Saat membuka file CSV di perangkat lunak seperti Microsoft Excel, pastikan pengaturan pemisah desimal Anda dikonfigurasi untuk menggunakan **titik (.)** dan bukan koma (,). Jika tidak, angka desimal pada kolom 'Luas (m2)' dan 'Luas (ha)' mungkin akan salah diinterpretasikan atau hilang.
*   `output/visualization/`: Grafik perbandingan dan pie chart tutupan lahan (`grafik_perbandingan_luas.png`, `pie_tutupan_lahan_awal.png`, `pie_tutupan_lahan_akhir.png`), heatmap perubahan kelas (`heatmap_perubahan_kelas.png`), dan peta perubahan statis (`peta_perubahan_statis.png`). Untuk Mode 2, output akan berada di `output/visualization/new_area_visualization/`.
*   `output/change_map_new_area.tif`: Peta perubahan untuk area baru (Mode 2).
*   `output/log.txt`: File ini akan berisi log proses dari proyek ini, sehingga jika terjadi sebuah kesalahan akan mudah untuk menemukan pada tahapan mana kesalahan tersebut terjadi.

## Dependensi

Proyek ini dibangun menggunakan pustaka Python berikut (terdaftar di `requirements.txt`):

*   `pyyaml`
*   `rasterio`
*   `geopandas`
*   `shapely`
*   `numpy`
*   `scikit-learn`
*   `joblib`
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `pyproj`
*   `fiona` 