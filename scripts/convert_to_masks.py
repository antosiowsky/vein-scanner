import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Ścieżki
xml_path = "annotations.xml"
img_dir = "images/"
mask_dir = "masks_multiclass/"
os.makedirs(mask_dir, exist_ok=True)

# Parsuj XML
tree = ET.parse(xml_path)
root = tree.getroot()

for image_tag in root.findall(".//image"):
    file_name = image_tag.get("name")
    image_path = os.path.join(img_dir, file_name)

    # Sprawdź czy plik obrazu istnieje
    if not os.path.exists(image_path):
        print(f"[⚠️] Brak pliku obrazu: {file_name}")
        continue

    # Sprawdź czy są anotacje
    polyline_tags = image_tag.findall("polyline")
    polygon_tags = image_tag.findall("polygon")
    if not polyline_tags and not polygon_tags:
        print(f"[ℹ️] Brak anotacji dla {file_name}")
        continue

    width = int(image_tag.get("width"))
    height = int(image_tag.get("height"))
    mask = np.zeros((height, width, 3), dtype=np.uint8)  # 3-kanałowa maska
    colorR = (0, 0, 255)      # Czerwony (BGR)
    colorY = (0, 255, 255)    # Żółty (BGR)

    # Rysuj rękę (label="arm") jako czerwoną
    for polygon_tag in polygon_tags:
        if polygon_tag.get("label") != "arm":
            continue
        points = np.array([
            [float(x), float(y)] for x, y in
            [p.split(",") for p in polygon_tag.get("points").split(";")]
        ], np.int32)
        cv2.fillPoly(mask, [points], colorR)

    # Rysuj żyły (label="vein") jako żółte — nadpisują rękę
    for polyline_tag in polyline_tags:
        if polyline_tag.get("label") != "vein":
            continue
        points = np.array([
            [float(x), float(y)] for x, y in
            [p.split(",") for p in polyline_tag.get("points").split(";")]
        ], np.int32)
        if len(points) >= 2:
            cv2.polylines(mask, [points], isClosed=False, color=colorY, thickness=3)

    # Zapis maski
    mask_filename = os.path.splitext(file_name)[0] + "_mask.png"
    cv2.imwrite(os.path.join(mask_dir, mask_filename), mask)
    print(f"[✅] Zapisano maskę wieloklasową: {mask_filename}")

