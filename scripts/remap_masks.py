import os
import cv2
import numpy as np
from tqdm import tqdm

# --- KONFIGURACJA REMAPOWANIA (WYPEŁNIONA POPRAWNYMI WARTOŚCIAMI) ---
ORIGINAL_HAND_VALUE = 76  # Wartość dla ręki znaleziona w Twoich danych
ORIGINAL_VEIN_VALUE = 225 # Wartość dla żył znaleziona w Twoich danych

FOLDERS_TO_REMAP = ["augmentowane_dane/masks", "test_data/masks"]

def remap_mask_values(folder_path):
    print(f"\n--- Remapowanie wartości w folderze: '{folder_path}' ---")
    if not os.path.isdir(folder_path):
        return

    all_files = os.listdir(folder_path)
    for filename in tqdm(all_files, desc=f"Remapowanie masek"):
        mask_path = os.path.join(folder_path, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        new_mask = np.zeros_like(mask)
        # Remapowanie: stara wartość -> nowa wartość
        new_mask[mask == ORIGINAL_HAND_VALUE] = 1  # Ręka
        new_mask[mask == ORIGINAL_VEIN_VALUE] = 2  # Żyły
        
        # Nadpisz plik nową, poprawną maską
        cv2.imwrite(mask_path, new_mask)

if __name__ == "__main__":
    print("--- Rozpoczynam proces remapowania wartości masek ---")
    print(f"Ręka: {ORIGINAL_HAND_VALUE} -> 1")
    print(f"Żyły: {ORIGINAL_VEIN_VALUE} -> 2")
    print("Wszystko inne -> 0 (tło)")
    
    for folder in FOLDERS_TO_REMAP:
        remap_mask_values(folder)
        
    print("\n--- Zakończono remapowanie ---")
    print("Teraz maski powinny zawierać tylko wartości 0, 1, 2.")