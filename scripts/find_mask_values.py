import os
import cv2
import numpy as np
from tqdm import tqdm

# Podaj folder z ORYGINALNYMI, "brudnymi" maskami
FOLDER_TO_ANALYZE = "augmentowane_dane/masks" 

def find_unique_values(folder_path):
    print(f"--- Analiza unikalnych wartości w folderze: '{folder_path}' ---")
    if not os.path.isdir(folder_path):
        print(f"BŁĄD: Folder nie istnieje.")
        return

    # Zbieramy wszystkie unikalne wartości ze wszystkich masek
    total_unique_values = set()
    all_files = os.listdir(folder_path)

    for filename in tqdm(all_files, desc="Analiza masek"):
        mask_path = os.path.join(folder_path, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_in_file = np.unique(mask)
            total_unique_values.update(unique_in_file)
            
    print("\n--- Zakończono analizę ---")
    print(f"Znaleziono następujące unikalne wartości we wszystkich maskach: {sorted(list(total_unique_values))}")
    print("\nNa podstawie tych wartości musisz zdecydować, która odpowiada ręce, a która żyłom.")

if __name__ == "__main__":
    find_unique_values(FOLDER_TO_ANALYZE)