import folium
import rasterio
import numpy as np
import os
from rasterio.plot import reshape_as_image
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from matplotlib import colors
import tempfile

def generate_color_map():
    return {
        0: "#d73027",  # Non-Vegetasi: Merah
        1: "#fee08b",  # Vegetasi Sedang: Kuning
        2: "#1a9850"   # Vegetasi Tinggi: Hijau
    }

def raster_to_image(raster_path, colormap, output_image_path):
    with rasterio.open(raster_path) as src:
        array = src.read(1, resampling=Resampling.nearest)
        array = np.where(array == src.nodata, np.nan, array)

        # Buat RGB image dari colormap
        rgb = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
        cmap = colors.ListedColormap([colormap[i] for i in sorted(colormap.keys())])
        norm = colors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=3)

        plt.imsave(output_image_path, array, cmap=cmap, norm=norm)

        bounds = src.bounds
        return [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]  # SouthWest, NorthEast

def visualize_raster_interactive(raster_path, output_html="outputs/map.html"):
    colormap = generate_color_map()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image_path = tmp.name

    bounds = raster_to_image(raster_path, colormap, image_path)

    # Buat peta interaktif
    m = folium.Map(location=[(bounds[0][0] + bounds[1][0]) / 2,
                             (bounds[0][1] + bounds[1][1]) / 2],
                   zoom_start=12, tiles='OpenStreetMap')

    folium.raster_layers.ImageOverlay(
        name="Klasifikasi Lahan",
        image=image_path,
        bounds=bounds,
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"✅ Peta disimpan di {output_html}")
