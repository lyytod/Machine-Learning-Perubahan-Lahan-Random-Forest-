import os
import yaml
import numpy as np
import rasterio

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def ndvi_to_class(ndvi_array, nodata_value=None):
    ndvi = np.where(ndvi_array == nodata_value, np.nan, ndvi_array) if nodata_value is not None else ndvi_array.copy()
    labels = np.zeros_like(ndvi, dtype=np.uint8)
    labels[(ndvi > 0.2) & (ndvi <= 0.5)] = 1
    labels[ndvi > 0.5] = 2
    labels[np.isnan(ndvi)] = 255  # NoData
    return labels

def classify_and_save(ndvi_path, output_path):
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)
        meta = src.meta.copy()
        ndvi_nodata = src.nodata

    label_array = ndvi_to_class(ndvi, nodata_value=ndvi_nodata)
    meta.update(dtype=rasterio.uint8, count=1, nodata=255)

    with rasterio.open(output_path, "w", **meta) as dest:
        dest.write(label_array, 1)

    print(f"[✔] Labeled raster saved: {output_path}")

def main():
    config = load_config()
    os.makedirs("output", exist_ok=True)

    classify_and_save("output/ndvi_from_cropped.tif", "output/label_from.tif")
    classify_and_save("output/ndvi_to_cropped.tif", "output/label_to.tif")

if __name__ == "__main__":
    main()
