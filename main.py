# --- Parametry do łatwej podmiany ---
IMG_PATH = "anton1.png"           # np. "anton1.png", "obraz2.jpg"
CLAHE_CLIP_LIMIT = 12.0           # np. 2.0, 4.0, 8.0
CLAHE_TILE_GRID_SIZE = 12         # pojedyncza liczba, np. 8, 10, 16
MEDIAN_BLUR_KSIZE = 7             # np. 3, 5, 7 (musi być nieparzysty)
CLAHE_ITERATIONS = 1              # liczba iteracji CLAHE
# ------------------------------------

import cv2
import numpy as np

class ImageProcessing:
    @staticmethod
    def apply_clahe(frame_img, for_value, clip_limit, tile_grid_size, median_ksize):
        """Apply CLAHE processing to the given frame."""
        lab_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)

        for _ in range(int(for_value)):
            clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
            l_channel = clahe.apply(l_channel)

        lab_img = cv2.merge((l_channel, a_channel, b_channel))
        median_lab_img = cv2.medianBlur(lab_img, 3)

        return cv2.cvtColor(median_lab_img, cv2.COLOR_LAB2BGR)

img_gray = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# Image processing
processed_img = ImageProcessing.apply_clahe(
    frame_img=img_bgr,
    for_value=CLAHE_ITERATIONS,
    clip_limit=CLAHE_CLIP_LIMIT,
    tile_grid_size=CLAHE_TILE_GRID_SIZE,
    median_ksize=MEDIAN_BLUR_KSIZE
)

# Show results
resize_gray = cv2.resize(img_gray, (640, 360))
resize_img = cv2.resize(processed_img, (640, 360))

cv2.imshow("Original", resize_gray)
cv2.imshow("LAB CLAHE + Median", resize_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
