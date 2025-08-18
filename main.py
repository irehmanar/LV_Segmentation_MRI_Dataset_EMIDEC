import os
import time
import csv
from tqdm import tqdm

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ----------------------------
# Paths & setup
# ----------------------------
DATA_DIR = 'dataset'
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

os.makedirs('progress', exist_ok=True)

fieldnames = ['epoch','train_loss','val_loss','train_miou','val_miou','train_f1','val_f1','train_acc','val_acc']
with open(os.path.join('progress', 'UNet_vgg16.csv'), 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

print('the number of image/label in the train: ', len(os.listdir(x_train_dir)))
print('the number of image/label in the validation: ', len(os.listdir(x_valid_dir)))

# ----------------------------
# Albumentations pipelines
# ----------------------------
def get_training_augmentation():
    return A.Compose([
        A.Resize(256, 256, interpolation=cv2.INTER_LINEAR)
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(256, 256, interpolation=cv2.INTER_LINEAR)
    ])

def get_preprocessing():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=True)
    ])

# ----------------------------
# Dataset with GLOBAL mask remap
# ----------------------------
class Dataset(BaseDataset):
    """
    Reads images/masks, resizes, normalizes, and REMAPS mask pixel values to [0..C-1].
    The mapping is computed once per split by scanning all mask files so it's consistent.
    """
    # Names are for reference; loss only uses indices
    CLASSES = ['background', 'lv', 'myocardium', 'mi', 'mvo']

    def __init__(self, images_dir, masks_dir, classes, augmentation=None, preprocessing=None, custom_label_map=None):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps  = [os.path.join(masks_dir,  image_id) for image_id in self.ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # If you KNOW your palette, supply it as {raw_value: class_id} via custom_label_map
        # Example: {0:0, 50:1, 100:2, 150:3, 254:4}
        if custom_label_map is not None:
            self.value2id = dict(custom_label_map)
        else:
            self.value2id = self._build_global_label_map(self.masks_fps, max_classes=len(classes))

        # Invert to show in logs if needed
        self.id2value = {v: k for k, v in self.value2id.items()}

        # Sanity message
        print(f"[LabelMap] Raw→ID mapping (up to {len(classes)} classes): {self.value2id}")

    @staticmethod
    def _scan_mask_values(mask_paths, limit=2000000):
        """
        Collect unique grayscale values across all masks (capped for safety).
        """
        uniq = set()
        counted = 0
        for p in mask_paths:
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            vals = np.unique(m)
            uniq.update(int(v) for v in vals.tolist())
            counted += 1
            if counted >= limit:
                break
        return sorted(list(uniq))

    def _build_global_label_map(self, mask_paths, max_classes=5):
        """
        Build a consistent raw_value→class_id map across the split.
        Rule:
          - Ensure 0 maps to 0 (background)
          - For remaining raw values, sort ascending and map to 1.. up to max_classes-1
          - If more distinct non-zero values are present than allowed, keep the first (lowest) ones
            and map the extra values to background (0) with a warning.
        """
        uniq_vals = self._scan_mask_values(mask_paths)
        if 0 not in uniq_vals:
            uniq_vals = [0] + uniq_vals  # ensure background 0 exists

        non_zero = [v for v in uniq_vals if v != 0]
        if len(non_zero) == 0:
            # Degenerate case; only background
            return {0: 0}

        if len(non_zero) > (max_classes - 1):
            print(f"[WARN] Found {len(non_zero)} distinct non-zero mask values {non_zero} "
                  f"but only {max_classes} classes. "
                  f"Will map the smallest {max_classes-1} to 1.. and send the rest to background (0).")
            keep = non_zero[:max_classes - 1]
            drop = non_zero[max_classes - 1:]
        else:
            keep = non_zero
            drop = []

        value2id = {0: 0}
        for idx, v in enumerate(sorted(keep), start=1):
            value2id[v] = idx
        for v in drop:
            value2id[v] = 0  # map extras to background

        return value2id

    def _remap_mask_ids(self, mask_gray):
        """
        Vectorized remap using self.value2id; unknown values -> background 0.
        """
        lut = np.zeros(256, dtype=np.uint8)
        for raw_v, cls_id in self.value2id.items():
            lut[int(raw_v)] = int(cls_id)
        # unknown raw values remain 0
        return cv2.LUT(mask_gray, lut).astype(np.int64)

    def __getitem__(self, i):
        # read RGB image and GRAY mask
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_gray = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise RuntimeError(f"Failed to read mask: {self.masks_fps[i]}")

        # ---- REMAP raw values -> class ids [0..C-1]
        mask = self._remap_mask_ids(mask_gray)  # int64

        # spatial transforms
        if self.augmentation:
            transformed = self.augmentation(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        # normalization & tensor conversion
        if self.preprocessing:
            transformed = self.preprocessing(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        # dtypes
        mask = mask.long()
        return image, mask

    def __len__(self):
        return len(self.ids)

# ----------------------------
# Metrics
# ----------------------------
def pixel_accuracy(output, mask):
    with torch.no_grad():
        preds = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = (preds == mask).float().sum()
        total   = torch.numel(mask)
        return float(correct / total)

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=5):
    with torch.no_grad():
        probs = F.softmax(pred_mask, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds_f = preds.contiguous().view(-1)
        mask_f  = mask.contiguous().view(-1)

        # Confusion matrix
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for i in range(n_classes):
            for j in range(n_classes):
                cm[i, j] = torch.logical_and(mask_f == i, preds_f == j).sum().item()

        # F1 per class
        f1_scores = []
        for c in range(n_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            precision = tp / (tp + fp + 1e-10)
            recall    = tp / (tp + fn + 1e-10)
            f1        = 2 * precision * recall / (precision + recall + 1e-10)
            f1_scores.append(f1)

        # IoU per class
        ious = []
        for c in range(n_classes):
            pred_c = (preds_f == c)
            true_c = (mask_f  == c)
            inter  = torch.logical_and(pred_c, true_c).sum().float().item()
            union  = torch.logical_or(pred_c, true_c).sum().float().item()
            if union == 0:
                ious.append(np.nan)
            else:
                ious.append((inter + smooth) / (union + smooth))

        return np.nanmean(ious), float(np.nanmean(f1_scores))

def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg['lr']

# ----------------------------
# Training loop
# ----------------------------
def fit(epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, device, n_classes, patience=20, patch=False):
    torch.cuda.empty_cache()

    train_losses, val_losses = [], []
    train_iou, val_iou = [], []
    train_f1,  val_f1  = [], []
    train_acc, val_acc = [], []
    lrs = []

    best_val_loss = float('inf')
    best_val_miou = -float('inf')
    not_improve = 0

    batchsummary = {k: 0 for k in fieldnames}

    model.to(device)
    fit_time = time.time()

    for e in range(epochs):
        since = time.time()
        model.train()

        running_loss = 0.0
        epoch_iou = 0.0
        epoch_f1  = 0.0
        epoch_acc = 0.0

        for i, (image_tiles, mask_tiles) in enumerate(tqdm(train_loader)):
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()
                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles  = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device, non_blocking=True)
            mask  = mask_tiles.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(image)
            # Safety: ensure mask indices are valid
            if mask.max().item() >= n_classes or mask.min().item() < 0:
                raise RuntimeError(f"Mask contains invalid class ids (min={mask.min().item()}, max={mask.max().item()}) for n_classes={n_classes}")
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            miou_t, f1_t = mIoU(output, mask, n_classes=n_classes)
            epoch_iou += miou_t
            epoch_f1  += f1_t
            epoch_acc += pixel_accuracy(output, mask)
            lrs.append(get_lr(optimizer))

        mean_train_loss = running_loss / len(train_loader)
        mean_train_iou  = epoch_iou   / len(train_loader)
        mean_train_f1   = epoch_f1    / len(train_loader)
        mean_train_acc  = epoch_acc   / len(train_loader)

        # ---------------- Validation ----------------
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum  = 0.0
        val_f1_sum   = 0.0
        val_acc_sum  = 0.0

        with torch.no_grad():
            for (image_tiles, mask_tiles) in tqdm(val_loader):
                if patch:
                    bs, n_tiles, c, h, w = image_tiles.size()
                    image_tiles = image_tiles.view(-1, c, h, w)
                    mask_tiles  = mask_tiles.view(-1, h, w)

                image = image_tiles.to(device, non_blocking=True)
                mask  = mask_tiles.to(device, non_blocking=True)
                output = model(image)

                if mask.max().item() >= n_classes or mask.min().item() < 0:
                    raise RuntimeError(f"[VAL] Mask contains invalid class ids (min={mask.min().item()}, max={mask.max().item()}) for n_classes={n_classes}")

                loss = criterion(output, mask)
                val_loss_sum += loss.item()

                miou_v, f1_v = mIoU(output, mask, n_classes=n_classes)
                val_iou_sum  += miou_v
                val_f1_sum   += f1_v
                val_acc_sum  += pixel_accuracy(output, mask)

        mean_val_loss = val_loss_sum / len(val_loader)
        mean_val_iou  = val_iou_sum  / len(val_loader)
        mean_val_f1   = val_f1_sum   / len(val_loader)
        mean_val_acc  = val_acc_sum  / len(val_loader)

        # Record
        train_losses.append(mean_train_loss); val_losses.append(mean_val_loss)
        train_iou.append(mean_train_iou);    val_iou.append(mean_val_iou)
        train_f1.append(mean_train_f1);      val_f1.append(mean_val_f1)
        train_acc.append(mean_train_acc);    val_acc.append(mean_val_acc)

        print(f"Epoch:{e+1}/{epochs}.. "
              f"Train Loss: {mean_train_loss:.3f}.. "
              f"Val Loss: {mean_val_loss:.3f}.. "
              f"Train mIoU: {mean_train_iou:.3f}.. "
              f"Train F1: {mean_train_f1:.3f}.. "
              f"Val mIoU: {mean_val_iou:.3f}.. "
              f"Val F1: {mean_val_f1:.3f}.. "
              f"Train Acc: {mean_train_acc:.3f}.. "
              f"Val Acc: {mean_val_acc:.3f}.. "
              f"Time: {(time.time()-since)/60:.2f}m")

        # CSV row
        batchsummary.update({
            'epoch': e+1,
            'train_loss': mean_train_loss,
            'val_loss': mean_val_loss,
            'train_miou': mean_train_iou,
            'val_miou': mean_val_iou,
            'train_f1': mean_train_f1,
            'val_f1': mean_val_f1,
            'train_acc': mean_train_acc,
            'val_acc': mean_val_acc
        })
        with open(os.path.join('progress', 'UNet_vgg16.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)

        # Checkpointing
        improved = False
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            not_improve = 0
            torch.save(model.state_dict(), os.path.join('progress', 'UNet_vgg16_best_val_loss.pth'))
            print(f"✓ Saved best-by-loss (val_loss={best_val_loss:.4f})")
            improved = True
        else:
            not_improve += 1

        if mean_val_iou > best_val_miou:
            best_val_miou = mean_val_iou
            torch.save(model.state_dict(), os.path.join('progress', 'UNet_vgg16_best_mIoU.pth'))
            print(f"✓ Saved best-by-mIoU (val_mIoU={best_val_miou:.4f})")
            improved = True

        if not_improve >= patience:
            print(f"Val loss did not improve for {patience} epochs. Stopping.")
            break

    history = {
        'train_loss': train_losses, 'val_loss': val_losses,
        'train_miou': train_iou,    'val_miou':  val_iou,
        'train_f1':   train_f1,     'val_f1':    val_f1,
        'train_acc':  train_acc,    'val_acc':   val_acc,
        'lrs': lrs
    }
    print('Total time: {:.2f} m'.format((time.time()-fit_time)/60))
    return history

# ----------------------------
# Model, optimizer, scheduler
# ----------------------------
CLASSES = ['Background', 'LV', 'Myocardium', 'MI', 'MVO']
NUM_CLASSES = len(CLASSES)
model = smp.Unet('vgg16', encoder_weights='imagenet', classes=NUM_CLASSES, activation=None)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# If you know your exact palette, you can pass custom_label_map to Dataset
# Example: custom_label_map = {0:0, 50:1, 100:2, 150:3, 254:4}
custom_label_map = None

train_dataset = Dataset(
    x_train_dir, y_train_dir, classes=CLASSES,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(),
    custom_label_map=custom_label_map
)
valid_dataset = Dataset(
    x_valid_dir, y_valid_dir, classes=CLASSES,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(),
    custom_label_map=custom_label_map
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=(device.type=='cuda'))
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=(device.type=='cuda'))

criterion = nn.CrossEntropyLoss()
max_lr = 1e-3
epochs = 100
weight_decay = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

# ----------------------------
# Train
# ----------------------------
history = fit(epochs, model, train_loader, valid_loader, criterion, optimizer, sched, device=device, n_classes=NUM_CLASSES)

# ----------------------------
# Plots
# ----------------------------
def plot_loss(history):
    plt.figure()
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch'); plt.ylabel('loss'); plt.xlabel('epoch')
    plt.legend(); plt.grid(True)
    plt.savefig('UNet_vgg16_Loss.png', bbox_inches='tight')
    plt.show()

def plot_score(history):
    plt.figure()
    plt.plot(history['train_miou'], label='train_mIoU', marker='*')
    plt.plot(history['val_miou'], label='val_mIoU', marker='*')
    plt.title('mIoU per epoch'); plt.ylabel('mean IoU'); plt.xlabel('epoch')
    plt.legend(); plt.grid(True)
    plt.savefig('UNet_vgg16_mIoU.png', bbox_inches='tight')
    plt.show()

def plot_f1_score(history):
    plt.figure()
    plt.plot(history['train_f1'], label='train_f1', marker='*')
    plt.plot(history['val_f1'], label='val_f1', marker='*')
    plt.title('F1 per epoch'); plt.ylabel('F1'); plt.xlabel('epoch')
    plt.legend(); plt.grid(True)
    plt.savefig('UNet_vgg16_F1.png', bbox_inches='tight')
    plt.show()

def plot_acc(history):
    plt.figure()
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch'); plt.ylabel('Accuracy'); plt.xlabel('epoch')
    plt.legend(); plt.grid(True)
    plt.savefig('UNet_vgg16_Accuracy.png', bbox_inches='tight')
    plt.show()

plot_loss(history)
plot_score(history)
plot_acc(history)
plot_f1_score(history)
