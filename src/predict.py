import numpy as np
import rasterio
import joblib
from rasterio.transform import from_origin
from src.utils import mask_by_boundary
from src.ndvi_to_class import ndvi_to_class

def predict_land_cover(rgb_path, model_path, boundary_path, output_path):
    # Baca data RGB
    with rasterio.open(rgb_path) as src:
        rgb = src.read().transpose((1, 2, 0))  # shape: (H, W, C)
        meta = src.meta.copy()
        rgb_masked, mask_geom = mask_by_boundary(src, boundary_path)

    # Masking valid pixel
    valid_mask = ~np.any(rgb_masked == 0, axis=-1)
    X = rgb_masked[valid_mask]

    # Load model
    clf = joblib.load(model_path)

    # Prediksi
    print("🚀 Melakukan prediksi tutupan lahan...")
    y_pred = np.full(valid_mask.shape, 255, dtype=np.uint8)
    y_pred[valid_mask] = clf.predict(X)

    # Simpan ke raster
    meta.update({
        "count": 1,
        "dtype": "uint8"
    })
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(y_pred, 1)

    print(f"✅ Hasil prediksi disimpan ke: {output_path}")

if __name__ == "__main__":
    predict_land_cover(
        rgb_path="data/rgb.tif",
        model_path="models/random_forest.pkl",
        boundary_path="data/boundary.shp",
        output_path="output/klasifikasi_prediksi.tif"
    )
