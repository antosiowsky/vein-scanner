# --- Parametry do łatwej podmiany ---
IMG_PATH = "anton1.png"           # np. "anton1.png", "obraz2.jpg"
CLAHE_CLIP_LIMIT = 12.0            # np. 2.0, 4.0, 8.0
CLAHE_TILE_GRID_SIZE = (12, 12)   # np. (8,8), (10,10), (16,16)
MEDIAN_BLUR_KSIZE = 7              # np. 3, 5, 7 (musi być nieparzysty)
# ------------------------------------

import cv2
import numpy as np

# Wczytanie obrazu w skali szarości (np. z kamery mono)
img_gray = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

# Zamiana na obraz 3-kanałowy (potrzebne do LAB)
img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# Konwersja do przestrzeni LAB
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(img_lab)

# CLAHE na kanale L
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
l_clahe = clahe.apply(l)

# Median blur na kanale L
l_median = cv2.medianBlur(l_clahe, ksize=MEDIAN_BLUR_KSIZE)

# Połączenie kanałów i powrót do BGR
lab_clahe = cv2.merge([l_median, a, b])
bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Zamiana na odcienie czerwieni (mapowanie na colormap)
red_colormap = cv2.applyColorMap(l_median, cv2.COLORMAP_HOT)

# Wyświetlenie wyników
cv2.imshow("Oryginał", img_gray)
cv2.imshow("LAB CLAHE + Median", bgr_clahe)
cv2.imshow("Czerwony Colormap", red_colormap)
cv2.waitKey(0)
cv2.destroyAllWindows()

