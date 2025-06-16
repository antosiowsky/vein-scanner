import cv2
import numpy as np 

# CLAHE filter parameters
IMG_PATH = "anton1.png"
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = 8
MEDIAN_BLUR_KSIZE = 3
CLAHE_ITERATIONS = 4

# Gabor filter parameters
GABOR_KSIZE = 41
GABOR_SIGMA = 4.0
GABOR_LAMBDA = 10.0
GABOR_GAMMA = 0.5
GABOR_PSI = 0
GABOR_THETA_LIST = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
# ------------------------------------

class ImageProcessing:
    @staticmethod
    def apply_clahe(frame_img, for_value, clip_limit, tile_grid_size, median_ksize=3):
        """Apply CLAHE and median blur to the given frame."""
        lab_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)

        for _ in range(int(for_value)):
            clahe = cv2.createCLAHE(
                clipLimit=float(clip_limit),
                tileGridSize=(int(tile_grid_size), int(tile_grid_size))
            )
            l_channel = clahe.apply(l_channel)

        lab_img = cv2.merge((l_channel, a_channel, b_channel))
        median_lab_img = cv2.medianBlur(lab_img, median_ksize)

        return cv2.cvtColor(median_lab_img, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def apply_gabor_bank(gray_img, ksize, sigma, lambd, gamma, psi, theta_list):
        """Apply multiple Gabor filters, enhance, blur, and quantize the result to 5 grayscale levels."""
        accum = np.zeros_like(gray_img, dtype=np.float32)

        for theta in theta_list:
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray_img, cv2.CV_32F, kernel)
            accum += np.abs(filtered)

        # Normalize to 0–255
        accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX)
        result = accum.astype(np.uint8)

        # Enhance contrast
        result = cv2.equalizeHist(result)

        # Light blur
        result = cv2.GaussianBlur(result, (3, 3), sigmaX=1)

        # --- Reduce to 5 grayscale levels using k-means ---
        Z = result.reshape((-1, 1)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4  # Number of colors
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        result = quantized.reshape(result.shape)
        # ---------------------------------------------------

        return result


# Wczytanie obrazu
img_gray = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

# CLAHE + Median
processed_img = ImageProcessing.apply_clahe(
    frame_img=img_bgr,
    for_value=CLAHE_ITERATIONS,
    clip_limit=CLAHE_CLIP_LIMIT,
    tile_grid_size=CLAHE_TILE_GRID_SIZE,
    median_ksize=MEDIAN_BLUR_KSIZE
)

# Konwersja do skali szarości
processed_gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

# Filtry Gabora (bank filtrów)
gabor_combined = ImageProcessing.apply_gabor_bank(
    gray_img=processed_gray,
    ksize=GABOR_KSIZE,
    sigma=GABOR_SIGMA,
    lambd=GABOR_LAMBDA,
    gamma=GABOR_GAMMA,
    psi=GABOR_PSI,
    theta_list=GABOR_THETA_LIST
)

# Wyświetlanie
cv2.imshow("Original", cv2.resize(img_gray, (640, 360)))
cv2.imshow("CLAHE + Median", cv2.resize(processed_gray, (640, 360)))
cv2.imshow("Gabor Combined", cv2.resize(gabor_combined, (640, 360)))

cv2.waitKey(0)
cv2.destroyAllWindows()