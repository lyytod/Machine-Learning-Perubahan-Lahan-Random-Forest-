import os
import yaml
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS, Transformer
import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def _get_utm_epsg_code(lon, lat):
    utm_band = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + utm_band
    else:
        return 32700 + utm_band

def reproject_to_utm(input_path, output_dir, file_type):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"reprojected_{base_name}")

    if file_type == "raster":
        with rasterio.open(input_path) as src:
            if src.crs and src.crs.is_projected and src.crs.name.startswith("UTM"):
                logger.info(f"[*] Raster '{input_path}' already in UTM: {src.crs.name}. Skipping reprojection.")
                return input_path
            
            logger.info(f"[*] Reprojecting raster '{input_path}' from {src.crs.to_string() if src.crs else 'unknown CRS'} to UTM...")
            
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            
            # Transform center coordinates to WGS84 if source CRS is not WGS84
            if src.crs and src.crs.to_epsg() != 4326:
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                center_lon, center_lat = transformer.transform(center_lon, center_lat)

            utm_epsg = _get_utm_epsg_code(center_lon, center_lat)
            target_crs = CRS.from_epsg(utm_epsg)

            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'count': src.count,
                'nodata': src.nodata # Pertahankan nilai nodata dari sumber
            })

            # Reproject all bands at once
            destination_data = np.zeros((src.count, height, width), dtype=src.meta['dtype'])

            reproject(
                source=src.read(),  # Baca semua band sebagai array 3D
                destination=destination_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )

            with rasterio.open(output_path, 'w', **kwargs) as dst:
                dst.write(destination_data)

        logger.info(f"[✔] Raster reprojected and saved: {output_path}")
        return output_path

    elif file_type == "vector":
        gdf = gpd.read_file(input_path)
        if gdf.crs and gdf.crs.is_projected and gdf.crs.name.startswith("UTM"):
            logger.info(f"[*] Vector '{input_path}' already in UTM: {gdf.crs.name}. Skipping reprojection.")
            return input_path

        logger.info(f"[*] Reprojecting vector '{input_path}' from {gdf.crs.to_string() if gdf.crs else 'unknown CRS'} to UTM...")
        
        centroid = gdf.geometry.unary_union.centroid
        center_lon, center_lat = centroid.x, centroid.y
        
        # Transform center coordinates to WGS84 if source CRS is not WGS84
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(gdf.crs, "EPSG:4326", always_xy=True)
            center_lon, center_lat = transformer.transform(center_lon, center_lat)

        utm_epsg = _get_utm_epsg_code(center_lon, center_lat)
        target_crs = CRS.from_epsg(utm_epsg)

        gdf_reprojected = gdf.to_crs(target_crs)
        gdf_reprojected.to_file(output_path, driver="ESRI Shapefile")
        
        logger.info(f"[✔] Vector reprojected and saved: {output_path}")
        return output_path
    else:
        raise ValueError("Unsupported file type for reprojection. Must be 'raster' or 'vector'.")

def crop_raster_to_shapefile(raster_path, shapefile_path, output_path):
    gdf = gpd.read_file(shapefile_path)
    geometries = [mapping(geom) for geom in gdf.geometry]

    with rasterio.open(raster_path) as src:
        # Gunakan src.nodata jika ada, jika tidak, gunakan nilai default yang sesuai
        # Untuk citra RGB (biasanya uint8), 0 atau 255 sering digunakan sebagai nodata.
        # Untuk NDVI (biasanya float), NaN atau nilai di luar rentang -1 hingga 1.
        # Kita akan gunakan src.nodata jika ada, atau 0 sebagai default. Jika tipe data adalah float, kita bisa menggunakan np.nan.
        # Untuk lebih aman, kita bisa menggunakan nilai yang tidak mungkin muncul di data valid.
        nodata_value = src.nodata if src.nodata is not None else 0 # Default ke 0 jika src.nodata tidak ada
        if np.issubdtype(src.meta['dtype'], np.floating):
            nodata_value = np.nan # Gunakan NaN untuk tipe data float

        out_image, out_transform = mask(src, geometries, crop=True, filled=True, nodata=nodata_value)
        out_meta = src.meta.copy()

    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "crs": src.crs,
        "nodata": nodata_value # Pastikan nodata value juga diperbarui di metadata
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        # Pastikan out_image memiliki dimensi yang benar untuk dest.write
        # Jika out_image.ndim == 2 (untuk band tunggal), perlu diubah menjadi (1, H, W)
        if out_image.ndim == 2:
            dest.write(out_image[np.newaxis, :, :])
        else:
            dest.write(out_image)

    logger.info(f"[✔] Cropped saved: {output_path}")

def preprocess(ndvi_from_path, ndvi_to_path, rgb_path, boundary_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    reprojected_data_temp_dir = os.path.join(output_dir, "reprojected_temp")
    os.makedirs(reprojected_data_temp_dir, exist_ok=True)

    reprojected_ndvi_from_path = reproject_to_utm(ndvi_from_path, reprojected_data_temp_dir, "raster")
    reprojected_ndvi_to_path = reproject_to_utm(ndvi_to_path, reprojected_data_temp_dir, "raster")
    reprojected_rgb_path = reproject_to_utm(rgb_path, reprojected_data_temp_dir, "raster")
    reprojected_boundary_path = reproject_to_utm(boundary_path, reprojected_data_temp_dir, "vector")

    ndvi_from_clipped_path = os.path.join(output_dir, "ndvi_from_clipped.tif")
    ndvi_to_clipped_path = os.path.join(output_dir, "ndvi_to_clipped.tif")
    rgb_clipped_path = os.path.join(output_dir, "rgb_clipped.tif")

    crop_raster_to_shapefile(
        raster_path=reprojected_ndvi_from_path,
        shapefile_path=reprojected_boundary_path,
        output_path=ndvi_from_clipped_path
    )

    crop_raster_to_shapefile(
        raster_path=reprojected_ndvi_to_path,
        shapefile_path=reprojected_boundary_path,
        output_path=ndvi_to_clipped_path
    )

    crop_raster_to_shapefile(
        raster_path=reprojected_rgb_path,
        shapefile_path=reprojected_boundary_path,
        output_path=rgb_clipped_path
    )

    # Clean up temporary reprojected files (optional, can be uncommented later if needed)
    # for f in os.listdir(reprojected_data_temp_dir):
    #     os.remove(os.path.join(reprojected_data_temp_dir, f))
    # os.rmdir(reprojected_data_temp_dir)

    return ndvi_from_clipped_path, ndvi_to_clipped_path, rgb_clipped_path, reprojected_boundary_path

if __name__ == "__main__":
    print("This script is primarily intended to be called by main.py.")
    print("For standalone testing, uncomment the example usage block.")
