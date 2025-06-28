# -*- coding: utf-8 -*-

import os
import cv2
import albumentations as A
from tqdm import tqdm

### --- KONFIGURACJA --- ###
INPUT_IMAGE_DIR = "images"
INPUT_MASK_DIR = "masks"
OUTPUT_DIR = "augmentowane_dane"
NUM_AUGMENTATIONS_PER_IMAGE = 14
COPY_ORIGINALS = True
### --- KONIEC KONFIGURACJI --- ###


def get_augmentation_pipeline():
    """
    Definiuje potok augmentacji. WERSJA POPRAWIONA - bez ostrzeżeń.
    """
    return A.Compose([
        # Poprawione argumenty: 'fill_value' i 'mask_fill_value'
        A.Rotate(limit=15, p=0.8, border_mode=cv2.BORDER_CONSTANT, fill_value=0, mask_fill_value=0),

        A.HorizontalFlip(p=0.5),

        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),

        # Zastąpiono przestarzałe ShiftScaleRotate nowszym i bardziej elastycznym Affine
        A.Affine(
            translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            scale=(0.85, 1.15), # scale_limit=0.15 -> skala od 1-0.15 do 1+0.15
            rotate=0,
            p=0.8,
            cval=0,         # Wartość wypełnienia dla obrazu
            cval_mask=0     # Wartość wypełnienia dla maski
        ),
        
        # Poprawiony argument: 'variance_limit' zamiast 'var_limit'
        A.GaussNoise(variance_limit=(10.0, 50.0), p=0.3),
        
        A.Blur(blur_limit=3, p=0.2),
    ])


def augment_and_save_data():
    """
    Wczytuje dane, augmentuje je i zapisuje na dysku.
    """
    print("--- Rozpoczynam proces augmentacji (wersja poprawiona) ---")
    output_image_dir = os.path.join(OUTPUT_DIR, "images")
    output_mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    print(f"Dane wyjściowe będą zapisywane w: '{os.path.abspath(OUTPUT_DIR)}'")

    if not os.path.isdir(INPUT_IMAGE_DIR):
        print(f"BŁĄD: Folder z obrazami '{INPUT_IMAGE_DIR}' nie istnieje. Zakończono.")
        return

    pipeline = get_augmentation_pipeline()
    image_filenames = sorted(os.listdir(INPUT_IMAGE_DIR))
    total_files_processed = 0

    for filename in tqdm(image_filenames, desc="Augmentacja"):
        image_path = os.path.join(INPUT_IMAGE_DIR, filename)
        mask_path = os.path.join(INPUT_MASK_DIR, filename)

        if not os.path.exists(mask_path):
            print(f"\nOstrzeżenie: Brak maski dla obrazu '{filename}'. Pomijam.")
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"\nOstrzeżenie: Błąd wczytywania pary plików dla '{filename}'. Pomijam.")
            continue

        if COPY_ORIGINALS:
            cv2.imwrite(os.path.join(output_image_dir, filename), image)
            cv2.imwrite(os.path.join(output_mask_dir, filename), mask)

        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            augmented = pipeline(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            base_name, extension = os.path.splitext(filename)
            new_filename = f"{base_name}_aug_{i+1}{extension}"
            cv2.imwrite(os.path.join(output_image_dir, new_filename), aug_image)
            cv2.imwrite(os.path.join(output_mask_dir, new_filename), aug_mask)
        
        total_files_processed += 1

    total_generated = total_files_processed * NUM_AUGMENTATIONS_PER_IMAGE
    final_count = total_generated + (total_files_processed if COPY_ORIGINALS else 0)

    print("\n--- Zakończono augmentację ---")
    print(f"Przetworzono łącznie oryginalnych par obraz-maska: {total_files_processed}")
    print(f"Wygenerowano nowych par: {total_generated}")
    print(f"Łączna liczba par w folderze wyjściowym '{OUTPUT_DIR}': {final_count}")

if __name__ == "__main__":
    augment_and_save_data()