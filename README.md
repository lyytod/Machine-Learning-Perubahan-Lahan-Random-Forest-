# Deteksi Perubahan Tutupan Lahan Menggunakan Machine Learning

Proyek ini bertujuan untuk mendeteksi dan menganalisis perubahan tutupan lahan menggunakan citra satelit (NDVI dan RGB) dan teknik Machine Learning, serta menghasilkan visualisasi interaktif dari perubahan tersebut.

## Fitur Utama

*   **Preprocessing Citra:** Pemotongan (clipping) citra satelit (NDVI dan RGB) berdasarkan batas area studi.
*   **Klasifikasi NDVI:** Mengklasifikasikan nilai NDVI menjadi kategori tutupan lahan (Non-Vegetasi, Vegetasi Sedang, Vegetasi Tinggi).
*   **Deteksi Perubahan:** Mengidentifikasi dan menguantifikasi perubahan tutupan lahan antara dua periode waktu.
*   **Pelatihan & Prediksi Model:** Melatih model klasifikasi (Random Forest) menggunakan citra RGB dan label NDVI, kemudian menggunakannya untuk memprediksi tutupan lahan.
*   **Evaluasi Model:** Mengevaluasi kinerja model klasifikasi.
*   **Analisis Statistik:** Menghitung statistik area dan membuat matriks transisi perubahan tutupan lahan.
*   **Visualisasi:** Menghasilkan grafik statistik dan peta perubahan interaktif berbasis web.

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
│   └── boundary.shp
├── output/ (folder ini akan dihasilkan setelah menjalankan script)
│   ├── preprocessed/
│   ├── classified/
│   ├── model/
│   ├── prediction/
│   ├── evaluation/
│   ├── analysis/
│   ├── visualization/
│   └── peta_perubahan_interaktif.html
└── src/
    ├── analyze_change.py
    ├── change_detection.py
    ├── config.py
    ├── evaluate.py
    ├── generate_interactive_map.py
    ├── model.py
    ├── ndvi_to_class.py
    ├── predict.py
    ├── preprocessing.py
    └── utils.py
```

*   `config.yaml`: Berisi jalur input data dan jalur output untuk semua hasil yang dihasilkan.
*   `main.py`: Skrip utama yang mengorkestrasi seluruh alur kerja deteksi perubahan.
*   `requirements.txt`: Daftar pustaka Python yang dibutuhkan.
*   `data/`: Direktori untuk menyimpan data input mentah (citra NDVI, RGB, dan shapefile batas).
*   `output/`: Direktori tempat semua hasil pemrosesan, model, prediksi, analisis, dan visualisasi akan disimpan.
*   `src/`: Berisi modul-modul Python terpisah yang mengimplementasikan setiap langkah dalam alur kerja:
    *   `preprocessing.py`: Fungsi untuk memproses awal citra.
    *   `ndvi_to_class.py`: Klasifikasi nilai NDVI ke kelas tutupan lahan.
    *   `change_detection.py`: Deteksi perubahan antara dua peta klasifikasi.
    *   `model.py`: Ekstraksi fitur, pelatihan, dan penyimpanan model Machine Learning.
    *   `predict.py`: Melakukan prediksi tutupan lahan menggunakan model yang sudah dilatih.
    *   `evaluate.py`: Evaluasi kinerja model.
    *   `analyze_change.py`: Analisis statistik perubahan tutupan lahan.
    *   `visualization.py`: Generasi grafik statistik.
    *   `generate_interactive_map.py`: Pembuatan peta perubahan interaktif.
    *   `utils.py`: Fungsi-fungsi utilitas umum.
    *   `config.py`: Definisi nama kelas dan konstanta lainnya.

## Instalasi

Untuk menjalankan proyek ini, ikuti langkah-langkah berikut:

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

2.  **Jalankan Skrip Utama:**
    ```bash
    python main.py
    ```

    Skrip akan menjalankan seluruh pipeline secara berurutan, mulai dari preprocessing hingga pembuatan peta interaktif.

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

outputs:
  preprocessed: output/preprocessed
  classified: output/classified
  model: output/model
  prediction: output/prediction/prediction.tif
  evaluation: output/evaluation
  analysis: output/analysis
  visualization: output/visualization
  interactive_map: output/peta_perubahan_interaktif.html
```

## Output

Setelah eksekusi berhasil, folder `output/` akan berisi:

*   `output/preprocessed/`: Citra NDVI dan RGB yang telah dipotong.
*   `output/classified/`: Citra NDVI yang telah diklasifikasikan ke dalam kelas tutupan lahan.
*   `output/model/`: Model Machine Learning yang telah dilatih (`random_forest.pkl`).
*   `output/prediction/`: Peta prediksi tutupan lahan (`prediction.tif`).
*   `output/evaluation/`: Laporan evaluasi model.
*   `output/analysis/`: File CSV berisi statistik luas perubahan (`luas_perubahan.csv`) dan matriks transisi (`matrix_perubahan.csv`), serta statistik klasifikasi (`statistik_klasifikasi_from.csv`, `statistik_klasifikasi_to.csv`).
*   `output/visualization/`: Grafik perbandingan dan pie chart tutupan lahan (`grafik_perbandingan_luas.png`, `pie_tutupan_lahan_awal.png`, `pie_tutupan_lahan_akhir.png`), dan heatmap perubahan kelas (`heatmap_perubahan_kelas.png`).
*   `output/peta_perubahan_interaktif.html`: Peta interaktif yang dapat dibuka di browser web untuk memvisualisasikan perubahan tutupan lahan.

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
*   `folium`
*   `branca` 