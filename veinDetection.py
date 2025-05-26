import cv2
import numpy as np

# Wczytanie obrazu w skali szarości
img_gray = cv2.imread("anton1.png", cv2.IMREAD_GRAYSCALE)

# Zamiana na obraz 3-kanałowy (potrzebne do LAB)
img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# Konwersja do przestrzeni LAB
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(img_lab)

# CLAHE na kanale L
clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(10,10))
l_clahe = clahe.apply(l)

# Median blur na kanale L
l_median = cv2.medianBlur(l_clahe, ksize=3)

# Połączenie kanałów i powrót do BGR
lab_clahe = cv2.merge([l_median, a, b])
bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Wyświetlenie wyników
cv2.imshow("Oryginał", img_gray)
cv2.imshow("LAB CLAHE + Median", bgr_clahe)
cv2.waitKey(0)
cv2.destroyAllWindows()



"""# Progowanie obrazu (poziom 150)
threshold_value = 70

# Konwersja na obraz w skali szarości
gray_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_BGR2GRAY)

# Progowanie - wszystko poniżej threshold_value -> 0 (czarne), reszta -> 255 (białe)
_, thresholded_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)

# Wyświetlenie obrazu progowego
cv2.imshow("Thresholded Image", thresholded_img)
"""