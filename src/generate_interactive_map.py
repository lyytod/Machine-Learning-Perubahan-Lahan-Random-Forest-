# generate_interactive_map.py
import folium
import rasterio
import numpy as np
from rasterio.plot import reshape_as_image
from folium.raster_layers import ImageOverlay
from branca.element import Template, MacroElement

def create_interactive_map(tif_path):
    with rasterio.open(tif_path) as src:
        perubahan_data = src.read(1)
        bounds = src.bounds

    class_colors = {
        0: [0, 0, 0],         # 0->0 Non-Vegetasi -> Non-Vegetasi (Hitam)
        1: [255, 127, 14],    # 0->1 Non-Vegetasi -> Vegetasi Sedang (Jingga)
        2: [174, 199, 232],   # 0->2 Non-Vegetasi -> Vegetasi Tinggi (Biru Muda)
        10: [31, 119, 180],    # 1→0 Vegetasi Sedang → Non-Vegetasi (Biru Tua)
        11: [44, 160, 44],     # 1→1 Vegetasi Sedang → Vegetasi Sedang (Hijau)
        12: [148, 103, 189],   # 1→2 Vegetasi Sedang → Vegetasi Tinggi (Ungu)
        20: [227, 119, 194],   # 2→0 Vegetasi Tinggi → Non-Vegetasi (Pink)
        21: [188, 189, 34],    # 2→1 Vegetasi Tinggi → Vegetasi Sedang (Kuning kehijauan)
        22: [158, 218, 229]    # 2→2 Vegetasi Tinggi → Vegetasi Tinggi (Cyan)
    }

    rgb_image = np.zeros((perubahan_data.shape[0], perubahan_data.shape[1], 3), dtype=np.uint8)
    for class_value, color in class_colors.items():
        mask = perubahan_data == class_value
        for i in range(3):
            rgb_image[:, :, i][mask] = color[i]

    center_lat = (bounds.top + bounds.bottom) / 2
    center_lon = (bounds.left + bounds.right) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    img_overlay = ImageOverlay(
        image=rgb_image,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.7,
        name='Perubahan Lahan'
    )
    img_overlay.add_to(m)
    folium.LayerControl().add_to(m)

    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
         position: fixed;
         bottom: 50px;
         left: 50px;
         width: 160px;
         background-color: white;
         border: 2px solid grey;
         z-index: 9999;
         font-size: 14px;
         padding: 10px;
         font-family: sans-serif;">
      <strong>Perubahan</strong><br>
      <i style="background:#000000; width:15px; height:15px; float:left; margin-right:5px; opacity:1;"></i>Non-Vegetasi → Non-Vegetasi<br>
      <i style="background:#ff7f0e; width:15px; height:15px; float:left; margin-right:5px; opacity:1;"></i>Non-Vegetasi → Vegetasi Sedang<br>
      <i style="background:#aec7e8; width:15px; height:15px; float:left; margin-right:5px; opacity:1;"></i>Non-Vegetasi → Vegetasi Tinggi<br>
      <i style="background:#1f77b4; width:15px; height:15px; float:left; margin-right:5px; opacity:1;"></i>Vegetasi Sedang → Non-Vegetasi<br>
      <i style="background:#2ca02c; width:15px; height:15px; float:left; margin-right:5px; opacity:1;"></i>Vegetasi Sedang → Vegetasi Sedang<br>
      <i style="background:#9467bd; width:15px; height:15px; float:left; margin-right:5px; opacity:1;"></i>Vegetasi Sedang → Vegetasi Tinggi<br>
      <i style="background:#e377c2; width:15px; height:15px; float:left; margin-right:5px; opacity:1;"></i>Vegetasi Tinggi → Non-Vegetasi<br>
      <i style="background:#bcbd22; width:15px; height:15px; float:left; margin-right:5px; opacity:1;"></i>Vegetasi Tinggi → Vegetasi Sedang<br>
      <i style="background:#9edae5; width:15px; height:15px; float:left; margin-right:5px; opacity:1;"></i>Vegetasi Tinggi → Vegetasi Tinggi<br>
    </div>
    {% endmacro %}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    m.save("output/peta_perubahan_interaktif.html")
    print("Peta interaktif berhasil dibuat: output/peta_perubahan_interaktif.html")