# -*- coding: utf-8 -*-

import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import os

### --- KONFIGURACJA SKRYPTU (TUTAJ WPISZ SWOJE ŚCIEŻKI) --- ###

# 1. Ścieżka do Twojego wytrenowanego modelu
MODEL_PATH = "output_model/best_model.pth"

# 2. Ścieżka do obrazu, który chcesz przetestować
#    Pamiętaj, że ten obraz powinien być już po obróbce CLAHE i Median Filter.
INPUT_IMAGE_PATH = "save/images/20107_png.rf.8daf8614a8784e2202acbf1460e2954e.jpg" # <<< ZMIEŃ NA WŁAŚCIWĄ ŚCIEŻKĘ

# 3. Nazwa, pod jaką zostanie zapisany plik z kolorową maską
OUTPUT_IMAGE_PATH = "prediction_color_mask.png"

### --- KONFIGURACJA MODELU (musi być identyczna jak podczas treningu) --- ###
ENCODER = "mobilenet_v2"
NUM_CLASSES = 3
IMG_HEIGHT = 256
IMG_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

### --- FUNKCJE POMOCNICZE --- ###

def load_trained_model(model_path):
    """Wczytuje wytrenowany model i ustawia go w tryb ewaluacji."""
    print(f"Ładowanie modelu z: {model_path} na urządzenie: {DEVICE}")
    model = smp.Unet(encoder_name=ENCODER, in_channels=3, classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model załadowany pomyślnie.")
    return model

def create_color_mask(prediction_mask):
    """Tworzy kolorową maskę na podstawie predykcji klas."""
    # Definiujemy mapę kolorów (BGR, bo OpenCV tak zapisuje)
    # Tło (0) -> Czarny
    # Ręka (1) -> Biały
    # Żyły (2) -> Zielony
    color_map = {
        0: [0, 0, 0],       # Czarny
        1: [255, 255, 255], # Biały
        2: [0, 255, 0]      # Zielony
    }
    
    # Tworzymy pusty, 3-kanałowy obraz
    h, w = prediction_mask.shape
    color_mask_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Wypełniamy kolorami na podstawie klas
    for class_id, color in color_map.items():
        color_mask_img[prediction_mask == class_id] = color
        
    return color_mask_img

def run_prediction():
    """Główna funkcja, która wykonuje cały proces."""
    # Sprawdzenie, czy pliki istnieją
    if not os.path.exists(MODEL_PATH):
        print(f"BŁĄD KRYTYCZNY: Plik modelu nie został znaleziony: {MODEL_PATH}")
        return
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"BŁĄD KRYTYCZNY: Plik wejściowy nie został znaleziony: {INPUT_IMAGE_PATH}")
        return

    # Krok 1: Załaduj model
    model = load_trained_model(MODEL_PATH)
    
    # Krok 2: Wczytaj i przygotuj obraz wejściowy
    print(f"Wczytywanie i przetwarzanie obrazu: {INPUT_IMAGE_PATH}")
    input_image = cv2.imread(INPUT_IMAGE_PATH)
    original_height, original_width, _ = input_image.shape
    
    transform = A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    input_tensor = transform(image=input_image)["image"].unsqueeze(0).to(DEVICE)
    
    # Krok 3: Uruchom predykcję
    print("Uruchamianie predykcji...")
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prediction_mask = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Krok 4: Przeskaluj maskę predykcji do oryginalnego rozmiaru
    resized_mask = cv2.resize(
        prediction_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Krok 5: Stwórz kolorową maskę
    color_mask_output = create_color_mask(resized_mask)
    
    # Krok 6: Zapisz wynik do pliku
    cv2.imwrite(OUTPUT_IMAGE_PATH, color_mask_output)
    print(f"Kolorowa maska została pomyślnie zapisana w: {OUTPUT_IMAGE_PATH}")
    
    # Krok 7: Wyświetl wizualizację porównawczą
    print("Wyświetlanie wyników... Naciśnij dowolny klawisz, aby zamknąć okno.")
    
    # Połącz oryginalny obraz i wynik w jeden duży obraz
    comparison_image = np.concatenate((input_image, color_mask_output), axis=1)
    
    # Dodaj etykiety tekstowe
    cv2.putText(comparison_image, 'Oryginal (po CLAHE)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(comparison_image, 'Wynik Segmentacji', (original_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Porownanie Wynikow", comparison_image)
    cv2.waitKey(0)  # Czekaj na naciśnięcie klawisza
    cv2.destroyAllWindows()

### --- URUCHOMIENIE --- ###
if __name__ == "__main__":
    run_prediction()