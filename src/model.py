import numpy as np
import rasterio
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.ndvi_to_class import ndvi_to_class
from src.utils import mask_by_boundary
import logging

logger = logging.getLogger(__name__)

def extract_features(rgb_path, ndvi_path, boundary_path):
    # Baca data RGB dan NDVI
    with rasterio.open(rgb_path) as rgb_src:
        rgb = rgb_src.read().transpose((1, 2, 0))  # shape: (H, W, C)
        rgb_masked, _, rgb_nodata = mask_by_boundary(rgb_src, boundary_path)

    with rasterio.open(ndvi_path) as ndvi_src:
        ndvi = ndvi_src.read(1)
        ndvi_masked, _, ndvi_nodata = mask_by_boundary(ndvi_src, boundary_path)
        # Tambahkan squeeze untuk memastikan ndvi_masked selalu 2D (H, W)
        if ndvi_masked.ndim == 3 and ndvi_masked.shape[2] == 1:
            ndvi_masked = ndvi_masked.squeeze(axis=2)

    # Mask NoData
    # Gunakan nilai nodata yang sebenarnya
    if rgb.ndim == 3:
        rgb_no_data_mask = ~np.all(rgb_masked == rgb_nodata, axis=-1) # Gunakan np.all jika ingin semua band nodata
    else:
        rgb_no_data_mask = ~(rgb_masked == rgb_nodata)

    # Untuk NDVI, pastikan nodata juga ditangani
    ndvi_no_data_mask = ~(ndvi_masked == ndvi_nodata)

    valid_mask = rgb_no_data_mask & ndvi_no_data_mask

    # Ambil fitur dan target
    X = rgb_masked[valid_mask]
    # Hapus baris berikut karena ndvi_masked sudah berisi label kelas yang benar
    # ndvi_class = ndvi_to_class(ndvi_masked, nodata_value=ndvi_nodata)
    # y = ndvi_class[valid_mask]
    
    # Gunakan ndvi_masked langsung sebagai target y setelah mask NoData
    y = ndvi_masked[valid_mask]

    return X, y

def train_and_save_model(X, y, model_path="models/random_forest.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log class distribution in training and test sets
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    logger.info(f"Distribusi kelas dalam set pelatihan: {dict(zip(unique_train, counts_train))}")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    logger.info(f"Distribusi kelas dalam set pengujian: {dict(zip(unique_test, counts_test))}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluasi
    logger.info("\n🧪 Evaluasi Model:")
    logger.info(classification_report(y_test, clf.predict(X_test), target_names=["Non-Vegetasi", "Vegetasi Sedang", "Vegetasi Tinggi"], labels=[0, 1, 2]))

    # Simpan model
    joblib.dump(clf, model_path)
    logger.info(f"📦 Model disimpan ke: {model_path}")

def train_and_predict(rgb_path, label_path, boundary_path, model_output, prediction_output):
    # Extract features and labels
    X, y = extract_features(rgb_path, label_path, boundary_path)

    # Train and save the model
    train_and_save_model(X, y, model_path=f"{model_output}/random_forest.pkl")

    # Perform prediction using the trained model
    from src.predict import predict_land_cover
    predict_land_cover(
        rgb_path=rgb_path,
        model_path=f"{model_output}/random_forest.pkl",
        output_path=prediction_output + "/prediction.tif"
    )

if __name__ == "__main__":
    # Contoh pemanggilan
    X, y = extract_features("data/rgb.tif", "data/ndvi_to.tif", "data/boundary.shp")
    train_and_save_model(X, y)
