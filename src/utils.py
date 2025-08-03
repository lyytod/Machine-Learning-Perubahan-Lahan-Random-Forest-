import rasterio
from rasterio.mask import mask
import fiona
import numpy as np
import logging
import os
import sys

def setup_logger(log_path="output/log.txt"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(stream=sys.stdout)
        ]
    )

def mask_by_boundary(raster_src, shapefile_path):
    """Memotong raster sesuai dengan batas shapefile"""
    with fiona.open(shapefile_path, "r") as shapefile:
        geoms = [feature["geometry"] for feature in shapefile]
    out_image, out_transform = mask(dataset=raster_src, shapes=geoms, crop=True)
    out_image = out_image.transpose((1, 2, 0)) if out_image.ndim == 3 else out_image[0]
    return out_image, out_transform, raster_src.nodata # Mengembalikan nodata_value

def normalize_image(img):
    """Normalisasi citra RGB ke rentang 0-1"""
    return img.astype(np.float32) / 255.0

def check_same_shape(raster1_path, raster2_path):
    """Pastikan dua raster memiliki bentuk yang sama"""
    with rasterio.open(raster1_path) as r1, rasterio.open(raster2_path) as r2:
        return r1.shape == r2.shape
