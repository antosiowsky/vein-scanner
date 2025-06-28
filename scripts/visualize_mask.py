import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Konfiguracja ---
MASK_TO_CHECK = "maskaTERAZ.png"  # <<< ZMIEŃ NAZWĘ PLIKU

# --- Główna logika ---
def visualize_single_mask(mask_path):
    print(f"Wizualizacja maski: {mask_path}")
    
    # Wczytaj maskę w skali szarości
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print("Błąd: Nie udało się wczytać pliku.")
        return

    # Sprawdź unikalne wartości
    unique_values = np.unique(mask)
    print(f"Znalezione unikalne wartości w masce: {unique_values}")
    
    # Stwórz kolorową wersję maski
    h, w = mask.shape
    color_viz = np.zeros((h, w, 3), dtype=np.uint8)
    color_viz[mask == 1] = [0, 255, 0]   # Zielony - ręka
    color_viz[mask == 2] = [255, 0, 0]   # Czerwony - żyły
    
    # Zapisz obraz jako PNG w oryginalnej rozdzielczości
    output_path = os.path.splitext(mask_path)[0] + "_visualization.png"
    success = cv2.imwrite(output_path, cv2.cvtColor(color_viz, cv2.COLOR_RGB2BGR))
    
    if success:
        print(f"Zapisano wizualizację do pliku: {output_path}")
    else:
        print("Błąd podczas zapisywania pliku.")

if __name__ == "__main__":
    visualize_single_mask(MASK_TO_CHECK)
