import numpy as np
import rasterio
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.ndvi_to_class import ndvi_to_class
from src.utils import mask_by_boundary

def extract_features(rgb_path, ndvi_path, boundary_path):
    # Baca data RGB dan NDVI
    with rasterio.open(rgb_path) as rgb_src:
        rgb = rgb_src.read().transpose((1, 2, 0))  # shape: (H, W, C)
        rgb_masked, _ = mask_by_boundary(rgb_src, boundary_path)

    with rasterio.open(ndvi_path) as ndvi_src:
        ndvi = ndvi_src.read(1)
        ndvi_masked, _ = mask_by_boundary(ndvi_src, boundary_path)

    # Mask NoData
    valid_mask = ~np.any(rgb_masked == 0, axis=-1) & (~np.isnan(ndvi_masked))

    # Ambil fitur dan target
    X = rgb_masked[valid_mask]
    ndvi_class = ndvi_to_class(ndvi_masked)
    y = ndvi_class[valid_mask]

    return X, y

def train_and_save_model(X, y, model_path="models/random_forest.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluasi
    print("\n🧪 Evaluasi Model:")
    print(classification_report(y_test, clf.predict(X_test), target_names=["Non-Vegetasi", "Vegetasi Sedang", "Vegetasi Tinggi"]))

    # Simpan model
    joblib.dump(clf, model_path)
    print(f"📦 Model disimpan ke: {model_path}")

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
        boundary_path=boundary_path,
        output_path=prediction_output + "/prediction.tif"
    )

if __name__ == "__main__":
    # Contoh pemanggilan
    X, y = extract_features("data/rgb.tif", "data/ndvi_to.tif", "data/boundary.shp")
    train_and_save_model(X, y)
