# src/evaluate_model.py

import numpy as np
import rasterio
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
# from .ndvi_to_class import ndvi_to_class # Hapus impor ini karena tidak lagi mengklasifikasikan ulang
from .config import CLASS_NAMES # Pastikan ini diimpor

def evaluate_model(predicted_path, ground_truth_ndvi_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    # Baca hasil prediksi
    with rasterio.open(predicted_path) as pred_src:
        pred = pred_src.read(1)
        pred_nodata = pred_src.nodata if pred_src.nodata is not None else 255

    # Baca ground truth labels (ini sudah diklasifikasikan dari step sebelumnya)
    with rasterio.open(ground_truth_ndvi_path) as gt_src:
        gt_labels = gt_src.read(1) # Langsung baca sebagai label
        gt_nodata = gt_src.nodata if gt_src.nodata is not None else 255

    # Filter NoData
    mask = (gt_labels != gt_nodata) & (pred != pred_nodata)
    y_true = gt_labels[mask].flatten()
    y_pred = pred[mask].flatten()

    # Evaluasi
    class_labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    report = classification_report(y_true, y_pred, target_names=class_labels, labels=[0, 1, 2], digits=4)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("=== Classification Report ===")
    print(report)
    print("Akurasi:", acc)

    # Simpan classification report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nAkurasi: {acc:.4f}")

    # Simpan confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Prediksi")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
