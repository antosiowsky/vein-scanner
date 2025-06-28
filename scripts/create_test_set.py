import os
import random
import shutil

# --- Konfiguracja ---
SOURCE_DIR = "augmentowane_dane"
TEST_DIR = "test_data"
TEST_SPLIT_RATIO = 0.10  # 10% danych pójdzie na testy

# --- Główna logika ---
def create_test_set(source_dir, test_dir, ratio):
    print("--- Tworzenie zbioru testowego ---")
    source_images = os.path.join(source_dir, "images")
    source_masks = os.path.join(source_dir, "masks")
    
    test_images = os.path.join(test_dir, "images")
    test_masks = os.path.join(test_dir, "masks")

    # Utwórz foldery docelowe
    os.makedirs(test_images, exist_ok=True)
    os.makedirs(test_masks, exist_ok=True)
    
    # Pobierz listę plików
    all_images = sorted(os.listdir(source_images))
    random.seed(42) # Ustawienie ziarna losowości dla powtarzalności
    random.shuffle(all_images)
    
    # Oblicz, ile plików przenieść
    num_to_move = int(len(all_images) * ratio)
    files_to_move = all_images[:num_to_move]
    
    print(f"Łączna liczba par: {len(all_images)}")
    print(f"Przenoszenie {len(files_to_move)} par do folderu '{test_dir}'...")
    
    moved_count = 0
    for filename in files_to_move:
        # Ścieżki źródłowe
        src_img_path = os.path.join(source_images, filename)
        src_mask_path = os.path.join(source_masks, filename)
        
        # Ścieżki docelowe
        dst_img_path = os.path.join(test_images, filename)
        dst_mask_path = os.path.join(test_masks, filename)
        
        # Przenieś pliki
        if os.path.exists(src_mask_path):
            shutil.move(src_img_path, dst_img_path)
            shutil.move(src_mask_path, dst_mask_path)
            moved_count += 1
        else:
            print(f"Ostrzeżenie: Brak maski dla '{filename}', pomijam przenoszenie.")
            
    print(f"\nZakończono. Przeniesiono {moved_count} par plików.")
    print(f"Teraz folder '{source_dir}' zawiera dane do treningu/walidacji, a '{test_dir}' do ostatecznych testów.")

if __name__ == "__main__":
    create_test_set(SOURCE_DIR, TEST_DIR, TEST_SPLIT_RATIO)