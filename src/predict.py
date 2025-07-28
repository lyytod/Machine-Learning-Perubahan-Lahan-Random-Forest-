import numpy as np
import rasterio
import joblib
# from rasterio.transform import from_origin # Remove unused import
# from src.utils import mask_by_boundary # Remove if rgb_path is already clipped
# from src.ndvi_to_class import ndvi_to_class # Remove unused import
import logging

logger = logging.getLogger(__name__)

# Remove boundary_path parameter as rgb_path is assumed to be clipped
def predict_land_cover(rgb_path, model_path, output_path):
    # Baca data RGB yang sudah terpotong
    with rasterio.open(rgb_path) as src:
        rgb_data = src.read().transpose((1, 2, 0))  # shape: (H, W, C)
        meta = src.meta.copy()
        rgb_nodata = src.nodata # Get nodata from the source

    # Masking valid pixel
    # Assuming rgb_data is (H, W, C), check all bands for nodata
    if rgb_data.ndim == 3:
        valid_mask = ~np.all(rgb_data == rgb_nodata, axis=-1)
    else: # Should not happen for RGB, but for robustness (if it's a single band image, but still check nodata)
        valid_mask = ~(rgb_data == rgb_nodata)

    X = rgb_data[valid_mask]

    # Load model
    clf = joblib.load(model_path)

    # Prediksi
    logger.info("🚀 Melakukan prediksi tutupan lahan...")
    y_pred_flat = clf.predict(X) # Predicted values for valid pixels

    # Rekonstruksi array prediksi penuh
    y_pred_full = np.full(valid_mask.shape, 255, dtype=np.uint8) # 255 for NoData
    y_pred_full[valid_mask] = y_pred_flat

    # Simpan ke raster
    meta.update({
        "count": 1,
        "dtype": "uint8",
        # Keep original transform, height, width, crs from the clipped RGB.
        # These are already correct from src.meta.
    })
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(y_pred_full, 1)

    logger.info(f"✅ Hasil prediksi disimpan ke: {output_path}")

if __name__ == "__main__":
    predict_land_cover(
        rgb_path="output/preprocessed/rgb_clipped.tif", # Example assumes preprocessed path
        model_path="output/model/random_forest.pkl", # Example assumes model output path
        # boundary_path="data/boundary.shp", # Removed
        output_path="output/prediction/prediction.tif"
    )
