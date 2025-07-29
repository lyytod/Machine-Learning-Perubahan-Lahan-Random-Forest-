# main.py
import yaml
from src.preprocessing import preprocess
from src.ndvi_to_class import classify_and_save
from src.model import train_and_predict
from src.evaluate import evaluate_model
from src.change_detection import detect_change
from src.analyze_change import compute_area_stats, save_stats_to_csv, compute_transition_matrix, plot_bar_comparison, plot_pie_chart, plot_transition_heatmap
from src.generate_static_map import generate_static_map
from src.predict import predict_land_cover
import logging
import rasterio
import os
import shutil # Import shutil for directory cleanup

def main():
    # Setup logging
    from src.utils import setup_logger
    setup_logger()
    logger = logging.getLogger(__name__)

    # Load configuration
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # --- Pilihan Mode ---
    print("Pilih mode operasi:")
    print("1. Jalankan Seluruh Pipeline (Termasuk Pelatihan Model Baru)")
    print("2. Prediksi Menggunakan Model yang Sudah Ada (untuk area lain)")
    choice = input("Masukkan pilihan (1 atau 2): ")

    if choice == '1':
        logger.info("[*] Mode 1: Menjalankan Seluruh Pipeline (dengan Pelatihan Model Baru).")
        run_full_pipeline(config, logger)
    elif choice == '2':
        logger.info("[*] Mode 2: Melakukan Prediksi Menggunakan Model yang Sudah Ada (untuk area lain).")
        run_prediction_with_existing_model(config, logger)
    else:
        logger.error("Pilihan tidak valid. Harap masukkan '1' atau '2'.")

def create_output_directories(config, logger, mode_2_specific=False):
    """Collects and creates all necessary output directories based on the config."""
    dirs_to_create = set()
    output_section = config["outputs"]
    
    # Add general output directories
    for output_key, output_path in output_section.items():
        if output_key.startswith("prediction_new_area") or output_key.startswith("change_map_new_area") or \
           output_key.startswith("analysis_new_area") or output_key.startswith("visualization_new_area"):
            if not mode_2_specific: # Only create new area specific dirs if in mode 2
                continue
        
        # For file paths, add their parent directory
        if '.' in os.path.basename(output_path):
            dirs_to_create.add(os.path.dirname(output_path))
        # For directory paths, add them directly
        else:
            dirs_to_create.add(output_path)
    
    for directory in dirs_to_create:
        if directory:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Direktori dibuat atau sudah ada: {directory}")

def run_full_pipeline(config, logger):
    create_output_directories(config, logger, mode_2_specific=False) # Create all general directories
    logger.info("[*] Memulai tahap Pra-pemrosesan...")
    ndvi_from_clipped, ndvi_to_clipped, rgb_clipped, boundary_reprojected = preprocess(
        ndvi_from_path=config["paths"]["ndvi_from"],
        ndvi_to_path=config["paths"]["ndvi_to"],
        rgb_path=config["paths"]["rgb"],
        boundary_path=config["paths"]["boundary"],
        output_dir=config["outputs"]["preprocessed"]
    )
    logger.info("[✔] Tahap Pra-pemrosesan selesai.")

    logger.info("[*] Memulai tahap Klasifikasi NDVI...")
    classify_and_save(
        ndvi_path=ndvi_from_clipped,
        output_path=os.path.join(config["outputs"]["classified"], "ndvi_class_from.tif")
    )
    classify_and_save(
        ndvi_path=ndvi_to_clipped,
        output_path=os.path.join(config["outputs"]["classified"], "ndvi_class_to.tif")
    )
    logger.info("[✔] Tahap Klasifikasi NDVI selesai.")

    logger.info("[*] Memulai tahap Deteksi Perubahan...")
    detect_change(
        label_from_path=os.path.join(config["outputs"]["classified"], "ndvi_class_from.tif"),
        label_to_path=os.path.join(config["outputs"]["classified"], "ndvi_class_to.tif"),
        output_raster_path=config["paths"]["change_map"],
        stats_csv_path=os.path.join(config["outputs"]["analysis"], "luas_perubahan.csv")
    )
    logger.info("[✔] Tahap Deteksi Perubahan selesai.")

    logger.info("[*] Memulai tahap Pelatihan dan Prediksi Model...")
    train_and_predict(
        rgb_path=rgb_clipped,
        label_path=os.path.join(config["outputs"]["classified"], "ndvi_class_to.tif"),
        boundary_path=boundary_reprojected,
        model_output=config["outputs"]["model"],
        prediction_output=config["outputs"]["prediction"]
    )
    logger.info("[✔] Tahap Pelatihan dan Prediksi Model selesai.")

    logger.info("[*] Memulai tahap Evaluasi...")
    evaluate_model(
        ground_truth_ndvi_path=os.path.join(config["outputs"]["classified"], "ndvi_class_to.tif"),
        predicted_path=config["outputs"]["prediction"],
        output_dir=config["outputs"]["evaluation"]
    )
    logger.info("[✔] Tahap Evaluasi selesai.")

    logger.info("[*] Memulai tahap Analisis...")
    with rasterio.open(os.path.join(config["outputs"]["classified"], "ndvi_class_from.tif")) as src_read:
        label_from = src_read.read(1)
        pixel_area = abs(src_read.transform[0] * src_read.transform[4])

    with rasterio.open(os.path.join(config["outputs"]["classified"], "ndvi_class_to.tif")) as src_read:
        label_to = src_read.read(1)

    stats_from, counts_from = compute_area_stats(label_from, pixel_area)
    stats_to, counts_to = compute_area_stats(label_to, pixel_area)

    save_stats_to_csv(stats_from, os.path.join(config["outputs"]["analysis"], "statistik_klasifikasi_from.csv"))
    save_stats_to_csv(stats_to, os.path.join(config["outputs"]["analysis"], "statistik_klasifikasi_to.csv"))

    transition_matrix = compute_transition_matrix(label_from, label_to)
    transition_matrix.to_csv(os.path.join(config["outputs"]["analysis"], "matrix_perubahan.csv"))

    plot_bar_comparison(stats_from, stats_to, config["outputs"]["visualization"])
    plot_pie_chart(stats_from, tahun="awal", output_dir=config["outputs"]["visualization"])
    plot_pie_chart(stats_to, tahun="akhir", output_dir=config["outputs"]["visualization"])
    plot_transition_heatmap(os.path.join(config["outputs"]["analysis"], "matrix_perubahan.csv"), config["outputs"]["visualization"])
    logger.info("[✔] Tahap Analisis selesai.")

    logger.info("[*] Memulai tahap Pembuatan Peta Statis...")
    generate_static_map(
        change_map_path=config["paths"]["change_map"],
        output_png_path=os.path.join(config["outputs"]["analysis"], "peta_perubahan_statis.png"),
        output_tif_path=os.path.join(config["outputs"]["analysis"], "peta_perubahan_statis.tif")
    )
    logger.info("[✔] Tahap Pembuatan Peta Statis selesai.")

    logger.info("Pipeline selesai dijalankan.")

def run_prediction_with_existing_model(config, logger):
    create_output_directories(config, logger, mode_2_specific=True) # Create specific new area directories

    rgb_from_new_path = config["paths"]["rgb_from_new_area"]
    rgb_to_new_path = config["paths"]["rgb_to_new_area"]
    boundary_new_path = config["paths"]["boundary_new_area"]
    model_path = os.path.join(config["outputs"]["model"], "random_forest.pkl")

    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model tidak ditemukan di: {model_path}. Harap jalankan 'Mode 1' terlebih dahulu atau pastikan model sudah ada.")
        return

    logger.info("[*] Memulai Pra-pemrosesan citra RGB untuk area baru...")
    # Preprocess new RGB images and boundary
    temp_preprocessed_dir = os.path.join(config["outputs"]["preprocessed"], "temp_new_area")
    os.makedirs(temp_preprocessed_dir, exist_ok=True)

    # Preprocess rgb_from_new_area
    logger.info(f"Memproses {rgb_from_new_path}...")
    _, _, rgb_from_new_clipped, boundary_reprojected_new = preprocess(
        ndvi_from_path=rgb_from_new_path, # Dummy, will be ignored by model but needed for preprocess signature
        ndvi_to_path=rgb_from_new_path,   # Dummy
        rgb_path=rgb_from_new_path,       # Actual RGB for prediction
        boundary_path=boundary_new_path,
        output_dir=os.path.join(temp_preprocessed_dir, "from")
    )
    
    # Preprocess rgb_to_new_area
    logger.info(f"Memproses {rgb_to_new_path}...")
    _, _, rgb_to_new_clipped, _ = preprocess( # _ means we don't care about boundary_reprojected_new from second call
        ndvi_from_path=rgb_to_new_path, # Dummy
        ndvi_to_path=rgb_to_new_path,   # Dummy
        rgb_path=rgb_to_new_path,       # Actual RGB for prediction
        boundary_path=boundary_new_path,
        output_dir=os.path.join(temp_preprocessed_dir, "to")
    )
    logger.info("[✔] Pra-pemrosesan citra RGB area baru selesai.")


    logger.info("[*] Memulai Prediksi Tutupan Lahan untuk Area Baru...")
    predict_land_cover(
        rgb_path=rgb_from_new_clipped,
        model_path=model_path,
        output_path=config["outputs"]["prediction_new_area_from"]
    )
    predict_land_cover(
        rgb_path=rgb_to_new_clipped,
        model_path=model_path,
        output_path=config["outputs"]["prediction_new_area_to"]
    )
    logger.info("[✔] Prediksi Tutupan Lahan Area Baru selesai.")

    logger.info("[*] Memulai Deteksi Perubahan untuk Area Baru...")
    detect_change(
        label_from_path=config["outputs"]["prediction_new_area_from"],
        label_to_path=config["outputs"]["prediction_new_area_to"],
        output_raster_path=config["outputs"]["change_map_new_area"],
        stats_csv_path=os.path.join(config["outputs"]["analysis_new_area"], "luas_perubahan_new_area.csv")
    )
    logger.info("[✔] Deteksi Perubahan Area Baru selesai.")

    logger.info("[*] Memulai Analisis Perubahan untuk Area Baru...")
    # Re-read the predicted labels for analysis
    with rasterio.open(config["outputs"]["prediction_new_area_from"]) as src_read:
        label_from = src_read.read(1)
        pixel_area = abs(src_read.transform[0] * src_read.transform[4]) # Get pixel area from clipped raster

    with rasterio.open(config["outputs"]["prediction_new_area_to"]) as src_read:
        label_to = src_read.read(1)
    
    # Compute and save area stats
    stats_from, counts_from = compute_area_stats(label_from, pixel_area)
    stats_to, counts_to = compute_area_stats(label_to, pixel_area)
    save_stats_to_csv(stats_from, os.path.join(config["outputs"]["analysis_new_area"], "statistik_klasifikasi_from_new_area.csv"))
    save_stats_to_csv(stats_to, os.path.join(config["outputs"]["analysis_new_area"], "statistik_klasifikasi_to_new_area.csv"))

    # Compute and save transition matrix
    transition_matrix = compute_transition_matrix(label_from, label_to)
    transition_matrix.to_csv(os.path.join(config["outputs"]["analysis_new_area"], "matrix_perubahan_new_area.csv"))

    # Plot visualizations
    plot_bar_comparison(stats_from, stats_to, config["outputs"]["visualization_new_area"])
    plot_pie_chart(stats_from, tahun="awal_new_area", output_dir=config["outputs"]["visualization_new_area"])
    plot_pie_chart(stats_to, tahun="akhir_new_area", output_dir=config["outputs"]["visualization_new_area"])
    plot_transition_heatmap(os.path.join(config["outputs"]["analysis_new_area"], "matrix_perubahan_new_area.csv"), config["outputs"]["visualization_new_area"])
    logger.info("[✔] Analisis Perubahan Area Baru selesai.")

    logger.info("[*] Memulai Pembuatan Peta Statis untuk Area Baru...")
    generate_static_map(
        change_map_path=config["outputs"]["change_map_new_area"],
        output_png_path=os.path.join(config["outputs"]["analysis_new_area"], "peta_perubahan_statis_new_area.png"),
        output_tif_path=os.path.join(config["outputs"]["analysis_new_area"], "peta_perubahan_statis_new_area.tif")
    )
    logger.info("[✔] Pembuatan Peta Statis Area Baru selesai.")
    
    # Optional: Clean up temporary preprocessed directory
    # shutil.rmtree(temp_preprocessed_dir)
    # logger.info(f"Direktori temporer dihapus: {temp_preprocessed_dir}")

    logger.info("Pipeline Prediksi dengan Model yang Sudah Ada selesai dijalankan.")

if __name__ == "__main__":
    main()
