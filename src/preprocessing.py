import os
import yaml
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def crop_raster_to_shapefile(raster_path, shapefile_path, output_path):
    # Load shapefile dan konversi geometri
    gdf = gpd.read_file(shapefile_path)
    geometries = [mapping(geom) for geom in gdf.geometry]

    # Open raster dan crop
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geometries, crop=True)
        out_meta = src.meta.copy()

    # Update metadata
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # Simpan raster hasil crop
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"[✔] Cropped saved: {output_path}")

def main():
    config = load_config()

    os.makedirs("output", exist_ok=True)

    crop_raster_to_shapefile(
        raster_path=config["paths"]["ndvi_from"],
        shapefile_path=config["paths"]["boundary"],
        output_path="output/ndvi_from_cropped.tif"
    )

    crop_raster_to_shapefile(
        raster_path=config["paths"]["ndvi_to"],
        shapefile_path=config["paths"]["boundary"],
        output_path="output/ndvi_to_cropped.tif"
    )

    crop_raster_to_shapefile(
        raster_path=config["paths"]["rgb"],
        shapefile_path=config["paths"]["boundary"],
        output_path="output/rgb_cropped.tif"
    )

if __name__ == "__main__":
    main()
