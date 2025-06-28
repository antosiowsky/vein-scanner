import os
import cv2
import numpy as np
from tqdm import tqdm

# --- Konfiguracja ---
FOLDERS_TO_CLEAN = ["augmentowane_dane/masks", "test_data/masks"]
BACKUP_DIR = "masks_backup" # Stworzy kopię zapasową oryginalnych masek

# --- Główna logika ---
def clean_mask_files(folder_path):
    """
    Czyści maski, zamieniając wszystkie wartości inne niż 1 i 2 na 0.
    """
    if not os.path.isdir(folder_path):
        print(f"Ostrzeżenie: Folder '{folder_path}' nie istnieje. Pomijam.")
        return

    # Stwórz folder na backup
    backup_folder = os.path.join(BACKUP_DIR, os.path.basename(folder_path))
    os.makedirs(backup_folder, exist_ok=True)

    print(f"\n--- Czyszczenie folderu: '{folder_path}' ---")
    
    cleaned_count = 0
    all_files = os.listdir(folder_path)

    for filename in tqdm(all_files, desc=f"Czyszczenie masek w '{folder_path}'"):
        mask_path = os.path.join(folder_path, filename)
        
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Stwórz kopię zapasową
            cv2.imwrite(os.path.join(backup_folder, filename), mask)
            
            # Stwórz nową, czystą maskę
            new_mask = np.zeros_like(mask)
            
            # Zachowaj tylko wartości 1 (ręka) i 2 (żyły)
            new_mask[mask == 1] = 1
            new_mask[mask == 2] = 2
            
            # Zapisz wyczyszczoną maskę, nadpisując starą
            cv2.imwrite(mask_path, new_mask)
            cleaned_count += 1
            
        except Exception as e:
            print(f"\nWystąpił błąd podczas czyszczenia pliku '{filename}': {e}")
    
    print(f"Wyczyszczono i utworzono kopię zapasową {cleaned_count} plików.")


if __name__ == "__main__":
    print("--- Rozpoczynam proces czyszczenia masek ---")
    print(f"Oryginalne pliki zostaną skopiowane do folderu '{BACKUP_DIR}'")
    
    for folder in FOLDERS_TO_CLEAN:
        clean_mask_files(folder)
        
    print("\n--- Zakończono czyszczenie ---")