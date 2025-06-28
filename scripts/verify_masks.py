import os
import cv2
import numpy as np
from tqdm import tqdm

# --- Konfiguracja ---
# Podaj folder, w którym znajdują się maski do sprawdzenia.
# Sprawdzimy zarówno dane treningowe, jak i testowe.
FOLDERS_TO_CHECK = ["augmentowane_dane/masks", "test_data/masks"]

# --- Główna logika ---
def verify_mask_values(folder_path):
    """
    Sprawdza unikalne wartości pikseli we wszystkich maskach w danym folderze.
    """
    if not os.path.isdir(folder_path):
        print(f"Ostrzeżenie: Folder '{folder_path}' nie istnieje. Pomijam.")
        return

    print(f"\n--- Sprawdzanie folderu: '{folder_path}' ---")
    
    problematic_files = []
    all_files = os.listdir(folder_path)

    for filename in tqdm(all_files, desc=f"Weryfikacja masek w '{folder_path}'"):
        mask_path = os.path.join(folder_path, filename)
        
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"\nNie udało się wczytać maski: {filename}")
                continue
            
            # Znajdź wszystkie unikalne wartości pikseli w masce
            unique_values = np.unique(mask)
            
            # Sprawdź, czy istnieją wartości inne niż 0, 1, 2
            invalid_values = [val for val in unique_values if val not in [0, 1, 2]]
            
            if invalid_values:
                problematic_files.append((filename, invalid_values))
                
        except Exception as e:
            print(f"\nWystąpił błąd podczas przetwarzania pliku '{filename}': {e}")
            
    return problematic_files


if __name__ == "__main__":
    print("--- Rozpoczynam audyt wartości w plikach masek ---")
    
    all_problematic_files = {}

    for folder in FOLDERS_TO_CHECK:
        problems = verify_mask_values(folder)
        if problems:
            all_problematic_files[folder] = problems

    print("\n" + "="*50)
    print(" RAPORT AUDYTU MASEK")
    print("="*50)

    if not all_problematic_files:
        print("\nWSZYSTKIE MASKI SĄ POPRAWNE! Gratulacje!")
    else:
        print("\nZidentyfikowano problemy w następujących plikach:")
        for folder, files in all_problematic_files.items():
            print(f"\nFolder: '{folder}'")
            for filename, values in files:
                print(f"  -> Plik: {filename} zawiera nieprawidłowe wartości: {values}")
        
        print("\nSUGESTIA: Użyj skryptu do czyszczenia masek lub ręcznie popraw te pliki.")
    
    print("\n--- Koniec audytu ---")