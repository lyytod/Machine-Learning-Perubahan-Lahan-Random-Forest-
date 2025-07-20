# src/evaluate_model.py

import numpy as np
import rasterio
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .ndvi_to_class import ndvi_to_class

def evaluate_model(predicted_path, ground_truth_ndvi_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # Baca hasil prediksi
    with rasterio.open(predicted_path) as pred_src:
        pred = pred_src.read(1)
        pred_nodata = pred_src.nodata if pred_src.nodata is not None else 255

    # Baca NDVI ground truth dan konversi ke label
    with rasterio.open(ground_truth_ndvi_path) as ndvi_src:
        ndvi = ndvi_src.read(1)
        ndvi_nodata = ndvi_src.nodata

    gt_labels = ndvi_to_class(ndvi, nodata_value=ndvi_nodata)

    # Filter NoData
    mask = (gt_labels != 255) & (pred != pred_nodata)
    y_true = gt_labels[mask].flatten()
    y_pred = pred[mask].flatten()

    # Evaluasi
    labels = ["Non-Vegetasi", "Vegetasi Sedang", "Vegetasi Tinggi"]
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("=== Classification Report ===")
    print(report)
    print("Akurasi:", acc)

    # Simpan classification report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nAkurasi: {acc:.4f}")

    # Simpan confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Prediksi")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
