import cv2
import numpy as np
import os

# --- Parametry do łatwej podmiany ---
INPUT_FOLDER = "zyly"
OUTPUT_FOLDER = "zylyOutput"
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = 8
MEDIAN_BLUR_KSIZE = 3
CLAHE_ITERATIONS = 4
# ------------------------------------

class ImageProcessing:
    @staticmethod
    def apply_clahe(frame_img, for_value, clip_limit, tile_grid_size, median_ksize):
        lab_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)

        for _ in range(int(for_value)):
            clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
            l_channel = clahe.apply(l_channel)

        lab_img = cv2.merge((l_channel, a_channel, b_channel))
        median_lab_img = cv2.medianBlur(lab_img, median_ksize)
        return cv2.cvtColor(median_lab_img, cv2.COLOR_LAB2BGR)

# Utwórz folder wyjściowy jeśli nie istnieje
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Przetwarzanie wszystkich plików w folderze wejściowym
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        img_gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"Nie można wczytać: {input_path}")
            continue
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        processed_img = ImageProcessing.apply_clahe(
            frame_img=img_bgr,
            for_value=CLAHE_ITERATIONS,
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=CLAHE_TILE_GRID_SIZE,
            median_ksize=MEDIAN_BLUR_KSIZE
        )

        cv2.imwrite(output_path, processed_img)
        print(f"Zapisano: {output_path}")
