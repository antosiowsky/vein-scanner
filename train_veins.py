# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp

### --- KONFIGURACJA --- ###

# 1. Ścieżki do danych po augmentacji i pre-processingu
DATA_DIR = "augmentowane_dane/"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
OUTPUT_DIR = "./output_model/"

# 2. Parametry modelu i treningu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ARCHITECTURE = "Unet"
ENCODER = "mobilenet_v2"
NUM_CLASSES = 3  # 0: tło, 1: ręka, 2: żyły

# 3. Hiperparametry treningu
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 50
IMG_HEIGHT = 256
IMG_WIDTH = 256
VALIDATION_SPLIT = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

### --- AUGMENTACJE I DATASET (WERSJA UPROSZCZONA) --- ###

# Transformacje nie zawierają już żadnego specjalnego pre-processingu.
# Jedyne, co robimy, to zmiana rozmiaru i normalizacja.
train_transform = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    # Możemy dodać lekkie augmentacje geometryczne, jeśli chcemy,
    # ale skoro dane są już zaugmentowane, nie jest to konieczne.
    # A.HorizontalFlip(p=0.5), 
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

class VeinDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # Wczytujemy obraz, który jest już po pre-processingu
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        return image, mask.long()

# Reszta kodu (get_loaders, check_metrics, main) jest identyczna jak w poprzednich wersjach
# i nie wymaga modyfikacji.
def get_loaders(image_dir, mask_dir, batch_size, train_transform, val_transform, val_split):
    dataset = VeinDataset(image_dir, mask_dir)
    
    num_samples = len(dataset)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=False)
    
    return train_loader, val_loader

def check_metrics(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    iou_per_class = [0] * NUM_CLASSES
    
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.softmax(model(x), dim=1)
            preds = torch.argmax(preds, dim=1)
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            
            for cls in range(NUM_CLASSES):
                pred_inds = (preds == cls)
                target_inds = (y == cls)
                intersection = (pred_inds & target_inds).sum()
                union = (pred_inds | target_inds).sum()
                iou_per_class[cls] += (intersection + 1e-6) / (union + 1e-6)

    num_batches = len(loader)
    iou_per_class = [iou / num_batches for iou in iou_per_class]
    
    accuracy = num_correct / num_pixels
    mean_iou = sum(iou_per_class) / NUM_CLASSES
    
    print(f"\nDokładność walidacji (pixel accuracy): {accuracy*100:.2f}%")
    print(f"Mean IoU walidacji: {mean_iou:.4f}")
    print(f"  - IoU dla TŁA (klasa 0):     {iou_per_class[0]:.4f}")
    print(f"  - IoU dla RĘKI (klasa 1):    {iou_per_class[1]:.4f}")
    print(f"  - IoU dla ŻYŁ (klasa 2):     {iou_per_class[2]:.4f}")
        
    model.train()
    return mean_iou

def main():
    print(f"Używane urządzenie: {DEVICE}")
    print("Przygotowywanie danych...")
    train_loader, val_loader = get_loaders(IMAGE_DIR, MASK_DIR, BATCH_SIZE, train_transform, val_transform, VALIDATION_SPLIT)
    print(f"Dane gotowe. Liczba próbek treningowych: {len(train_loader.dataset)}, walidacyjnych: {len(val_loader.dataset)}")
    
    print("Inicjalizacja modelu...")
    model = smp.Unet(encoder_name=ENCODER, in_channels=3, classes=NUM_CLASSES).to(DEVICE)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    
    best_iou = -1.0
    
    print("Rozpoczynanie treningu...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoka {epoch+1}/{NUM_EPOCHS} ---")
        
        model.train()
        loop = tqdm(train_loader, desc=f"Epoka {epoch+1} Trening")
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)
            
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                predictions = model(data)
                loss = loss_fn(predictions, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(loss=loss.item())
        
        current_iou = check_metrics(val_loader, model, device=DEVICE)
        
        if current_iou > best_iou:
            best_iou = current_iou
            print(f"==> Nowy najlepszy model! Mean IoU: {current_iou:.4f}. Zapisywanie...")
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
    
    print("\nTrening zakończony.")
    print(f"Najlepszy osiągnięty Mean IoU na zbiorze walidacyjnym: {best_iou:.4f}")
    print(f"Model zapisany w: {os.path.join(OUTPUT_DIR, 'best_model.pth')}")

if __name__ == "__main__":
    main()