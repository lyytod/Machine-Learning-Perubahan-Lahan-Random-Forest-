CLASS_NAMES = {
    0: "Non-Vegetasi",
    1: "Vegetasi Sedang",
    2: "Vegetasi Tinggi"
}

CLASS_MAPPING = {
    "Non-Vegetasi": 0,
    "Vegetasi Sedang": 1,
    "Vegetasi Tinggi": 2
}

# Definisi warna konsisten untuk setiap kelas tutupan lahan
CLASS_COLORS = {
    "Non-Vegetasi": "#FE7F0F",  # Orange
    "Vegetasi Sedang": "#1F77B4", # Biru
    "Vegetasi Tinggi": "#2DA02C"   # Hijau
}

# PIXEL_AREA_M2 = 100 # Assuming 10x10 meter resolution per pixel # Removed as calculated dynamically 