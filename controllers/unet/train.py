#!/home/prashanna/miniforge3/envs/rl/bin/python
"""
Train U-Net on collected + auto-labeled dataset with augmentation.

Reads:
  ../dataset/images/  (input images)
  ../dataset/masks/   (auto-generated labels)

Saves:
  unet_checkpoint.pth (best model by val IoU)
  unet_final.pth      (last epoch)
"""
import os
import sys
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_model import UNet

NUM_CLASSES = 5
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5
VAL_RATIO = 0.15
SEED = 42

CLASSES = {0: "soil", 1: "vegetation", 2: "obstacle", 3: "pedestrian", 4: "sky"}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_train_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Affine(
            translate_percent=(-0.1, 0.1), scale=(0.8, 1.2), rotate=(-15, 15),
            border_mode=cv2.BORDER_REFLECT, p=0.5
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.CLAHE(clip_limit=4.0, p=1),
            A.RandomGamma(gamma_limit=(70, 130), p=1),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=1),
        ], p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=6, p=1),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.OpticalDistortion(distort_limit=0.2, p=1),
        ], p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_augmentations():
    return A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, files, augmentations):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = files
        self.aug = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        mask = cv2.imread(os.path.join(self.mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        augmented = self.aug(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]

        return img, mask.long()


def compute_class_weights(img_dir, mask_dir, files):
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for fname in files:
        mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        for c in range(NUM_CLASSES):
            counts[c] += np.sum(mask == c)

    total = counts.sum()
    num_classes = NUM_CLASSES
    weights = total / (num_classes * counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def compute_iou(preds, labels, num_classes):
    preds = preds.view(-1)
    labels = labels.view(-1)
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        intersection = (pred_c & label_c).sum().float()
        union = (pred_c | label_c).sum().float()
        if union == 0:
            continue
        ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0


def compute_dice(preds, labels, num_classes):
    preds = preds.view(-1)
    labels = labels.view(-1)
    dice_per_class = []
    for c in range(num_classes):
        pred_c = preds == c
        label_c = labels == c
        intersection = (pred_c & label_c).sum().float()
        total = pred_c.sum().float() + label_c.sum().float()
        if total == 0:
            continue
        dice_per_class.append((2.0 * intersection / total).item())
    return np.mean(dice_per_class) if dice_per_class else 0.0


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs)
        loss = criterion(out, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs)
        loss = criterion(out, masks)
        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        total_iou += compute_iou(preds, masks, NUM_CLASSES) * imgs.size(0)
        total_dice += compute_dice(preds, masks, NUM_CLASSES) * imgs.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_iou / n, total_dice / n


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dataset')
    img_dir = os.path.join(base, 'images')
    mask_dir = os.path.join(base, 'masks')

    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    mask_files = set(os.listdir(mask_dir))
    paired = [f for f in all_images if f in mask_files and f != 'make_visible.py' and '_visible' not in f]
    print(f"Found {len(paired)} paired images (from {len(all_images)} total, {len(mask_files)} masks)")

    if len(paired) == 0:
        print("No paired data found. Run collect_data.py + label_data.py first.")
        return

    train_files, val_files = train_test_split(paired, test_size=VAL_RATIO, random_state=SEED)
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    train_ds = SegDataset(img_dir, mask_dir, train_files, get_train_augmentations())
    val_ds = SegDataset(img_dir, mask_dir, val_files, get_val_augmentations())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(device)

    class_weights = compute_class_weights(img_dir, mask_dir, train_files).to(device)
    print(f"Class weights: {dict(zip(CLASSES.values(), [f'{w:.3f}' for w in class_weights.cpu().numpy()]))}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_iou = 0.0
    ckpt_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\nStarting training for {EPOCHS} epochs (augmentation: 7 transforms on train)")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val IoU':>7} | {'Val Dice':>8} | {'LR':>10}")
    print("-" * 70)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {val_iou:6.4f} | {val_dice:8.4f} | {lr:10.2e}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_dice': val_dice,
            }, os.path.join(ckpt_dir, 'unet_checkpoint.pth'))

    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(ckpt_dir, 'unet_final.pth'))

    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Checkpoints saved to {ckpt_dir}")


if __name__ == '__main__':
    main()
