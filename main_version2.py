import os, time, csv
from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

# ----------------------------
# Config
# ----------------------------
DATA_DIR = 'dataset'
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')
x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

IMG_SIZE = 512
BATCH_SIZE = 4            # adjust for your GPU memory (start small for 512x512)
NUM_WORKERS = 2
CLASSES = ['Background', 'LV', 'Myocardium', 'MI', 'MVO']
NUM_CLASSES = len(CLASSES)
os.makedirs('progress', exist_ok=True)

CSV_PATH = os.path.join('progress', 'SwinUNet_512.csv')
fieldnames = ['epoch','train_loss','val_loss','train_miou','val_miou','train_f1','val_f1','train_acc','val_acc']
with open(CSV_PATH, 'w', newline='') as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ----------------------------
# Albumentations
# ----------------------------
def get_training_augmentation():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
    ])

def get_validation_augmentation():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
    ])

def get_preprocessing():
    return A.Compose([
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(transpose_mask=True)
    ])

# ----------------------------
# Dataset with global mask remap
# ----------------------------
class Dataset(BaseDataset):
    CLASSES = ['background', 'lv', 'myocardium', 'mi', 'mvo']

    def __init__(self, images_dir, masks_dir, classes, augmentation=None, preprocessing=None, custom_label_map=None):
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, i) for i in self.ids]
        self.masks_fps  = [os.path.join(masks_dir,  i) for i in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        if custom_label_map is not None:
            self.value2id = dict(custom_label_map)
        else:
            self.value2id = self._build_global_label_map(self.masks_fps, max_classes=len(classes))
        self.id2value = {v:k for k,v in self.value2id.items()}
        print(f"[LabelMap] Raw→ID mapping (up to {len(classes)} classes): {self.value2id}")

    @staticmethod
    def _scan_mask_values(mask_paths, limit=2000000):
        uniq = set()
        counted = 0
        for p in mask_paths:
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is None: continue
            uniq.update(int(v) for v in np.unique(m).tolist())
            counted += 1
            if counted >= limit:
                break
        return sorted(list(uniq))

    def _build_global_label_map(self, mask_paths, max_classes=5):
        uniq_vals = self._scan_mask_values(mask_paths)
        if 0 not in uniq_vals:
            uniq_vals = [0] + uniq_vals
        non_zero = [v for v in uniq_vals if v != 0]
        if len(non_zero) > (max_classes - 1):
            keep = non_zero[:max_classes-1]
            drop = non_zero[max_classes-1:]
            print(f"[WARN] too many classes, keeping {keep}, mapping {drop}→0")
        else:
            keep, drop = non_zero, []
        value2id = {0:0}
        for idx, v in enumerate(sorted(keep), start=1):
            value2id[v] = idx
        for v in drop:
            value2id[v] = 0
        return value2id

    def _remap_mask_ids(self, mask_gray):
        lut = np.zeros(256, dtype=np.uint8)
        for raw_v, cls_id in self.value2id.items():
            lut[int(raw_v)] = int(cls_id)
        return cv2.LUT(mask_gray, lut).astype(np.int64)

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_gray = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise RuntimeError(f"Failed to read mask: {self.masks_fps[i]}")

        mask = self._remap_mask_ids(mask_gray)

        if self.augmentation:
            transformed = self.augmentation(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        if self.preprocessing:
            transformed = self.preprocessing(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        mask = mask.long()
        return image, mask

    def __len__(self):
        return len(self.ids)

# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def pixel_accuracy(logits, mask):
    preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
    correct = (preds == mask).float().sum()
    total   = mask.numel()
    return float(correct / total)

@torch.no_grad()
def mIoU(logits, mask, smooth=1e-10, n_classes=NUM_CLASSES):
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    preds_f = preds.reshape(-1)
    mask_f  = mask.reshape(-1)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for i in range(n_classes):
        for j in range(n_classes):
            cm[i, j] = torch.logical_and(mask_f == i, preds_f == j).sum().item()
    f1_scores, ious = [], []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp + 1e-10)
        recall    = tp / (tp + fn + 1e-10)
        f1        = 2 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1)
        pred_c = (preds_f == c)
        true_c = (mask_f  == c)
        inter  = torch.logical_and(pred_c, true_c).sum().float().item()
        union  = torch.logical_or(pred_c, true_c).sum().float().item()
        ious.append((inter + smooth) / (union + smooth) if union > 0 else np.nan)
    return float(np.nanmean(ious)), float(np.nanmean(f1_scores))

def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg['lr']

# ----------------------------
# Losses: CE + Dice (multi-class)
# ----------------------------
class DiceLoss(nn.Module):
    def __init__(self, n_classes, smooth=1e-5):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.n_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        intersection = torch.sum(probs * targets_onehot, dims)
        cardinality  = torch.sum(probs + targets_onehot, dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1 - dice
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, n_classes, ce_weight=None, dice_weight=0.5, ce_weight_factor=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice = DiceLoss(n_classes)
        self.dice_w = dice_weight
        self.ce_w = ce_weight_factor
    def forward(self, logits, targets):
        return self.ce_w * self.ce(logits, targets) + self.dice_w * self.dice(logits, targets)

# ----------------------------
# Swin-UNet (2D) with deep supervision
# ----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(out_ch + skip_ch, out_ch)  # auto matches skip_ch
        self.conv2 = ConvBNReLU(out_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class SwinUNet(nn.Module):
    def __init__(self, encoder_name='swin_tiny_patch4_window7_224', num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            encoder_name,
            features_only=True,
            out_indices=(0,1,2,3),
            pretrained=True,
            img_size=IMG_SIZE
        )
        feats = self.backbone.feature_info
        c1_ch, c2_ch, c3_ch, c4_ch = feats.channels()
        self._enc_channels = (c1_ch, c2_ch, c3_ch, c4_ch)  # save for layout checks

        dec_ch = 256
        self.conv_c4 = ConvBNReLU(c4_ch, dec_ch)

        # project skips to a stable width
        self.proj3 = ConvBNReLU(c3_ch, dec_ch, k=1, s=1, p=0)
        self.proj2 = ConvBNReLU(c2_ch, dec_ch, k=1, s=1, p=0)
        self.proj1 = ConvBNReLU(c1_ch, dec_ch, k=1, s=1, p=0)

        self.up3 = UpBlock(dec_ch, dec_ch, dec_ch)
        self.up2 = UpBlock(dec_ch, dec_ch, dec_ch)
        self.up1 = UpBlock(dec_ch, dec_ch, dec_ch)

        self.head = nn.Conv2d(dec_ch, num_classes, kernel_size=1)
        self.aux3 = nn.Conv2d(dec_ch, num_classes, kernel_size=1)
        self.aux2 = nn.Conv2d(dec_ch, num_classes, kernel_size=1)

    @staticmethod
    def _to_nchw_with_known_c(t, expected_c):
        # If last dim is the channel dim and matches expected channels -> permute NHWC -> NCHW
        if t.ndim == 4 and t.shape[1] != expected_c and t.shape[-1] == expected_c:
            return t.permute(0, 3, 1, 2).contiguous()
        # If already NCHW (C==expected at dim=1), or something else, return as-is
        return t

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x)
        c1_ch, c2_ch, c3_ch, c4_ch = self._enc_channels

        # Enforce NCHW using the expected channel counts
        c1 = self._to_nchw_with_known_c(c1, c1_ch)
        c2 = self._to_nchw_with_known_c(c2, c2_ch)
        c3 = self._to_nchw_with_known_c(c3, c3_ch)
        c4 = self._to_nchw_with_known_c(c4, c4_ch)

        x  = self.conv_c4(c4)
        s3 = self.proj3(c3)
        d3 = self.up3(x,  s3)

        s2 = self.proj2(c2)
        d2 = self.up2(d3, s2)

        s1 = self.proj1(c1)
        d1 = self.up1(d2, s1)

        out_main = self.head(d1)
        out_aux3 = self.aux3(d3)
        out_aux2 = self.aux2(d2)
        return out_main, out_aux2, out_aux3





# ----------------------------
# Train / Val loops (deep supervision aware)
# ----------------------------
def up_to_size(logits, size_hw):
    return F.interpolate(logits, size=size_hw, mode='bilinear', align_corners=False)

def fit(epochs, model, train_loader, val_loader, criterion_main, criterion_aux, optimizer, scheduler, device, n_classes, aux_weights=(1.0, 0.4, 0.2), patience=20):
    torch.cuda.empty_cache()
    best_val_loss = float('inf')
    best_val_miou = -float('inf')
    not_improve = 0

    history = {'train_loss':[], 'val_loss':[],
               'train_miou':[], 'val_miou':[],
               'train_f1':[], 'val_f1':[],
               'train_acc':[], 'val_acc':[],
               'lrs': []}

    for e in range(epochs):
        since = time.time()
        model.train()
        run_loss = run_iou = run_f1 = run_acc = 0.0

        for imgs, masks in tqdm(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits_main, logits_d2, logits_d3 = model(imgs)

            # Upsample ALL logits to full mask size (512x512)
            full_size = masks.shape[-2:]
            logits_main_up = up_to_size(logits_main, full_size)
            logits_d2_up   = up_to_size(logits_d2,   full_size)
            logits_d3_up   = up_to_size(logits_d3,   full_size)

            loss = (aux_weights[0]*criterion_main(logits_main_up, masks) +
                    aux_weights[1]*criterion_aux (logits_d2_up,   masks) +
                    aux_weights[2]*criterion_aux (logits_d3_up,   masks))
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                run_loss += loss.item()
                miou, f1 = mIoU(logits_main_up, masks, n_classes=n_classes)
                run_iou += miou
                run_f1  += f1
                run_acc += pixel_accuracy(logits_main_up, masks)
                history['lrs'].append(get_lr(optimizer))

        mean_train_loss = run_loss/len(train_loader)
        mean_train_iou  = run_iou/len(train_loader)
        mean_train_f1   = run_f1/len(train_loader)
        mean_train_acc  = run_acc/len(train_loader)

        # ----------------- Validation -----------------
        model.eval()
        val_loss = val_iou = val_f1 = val_acc = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits_main, logits_d2, logits_d3 = model(imgs)

                full_size = masks.shape[-2:]
                logits_main_up = up_to_size(logits_main, full_size)
                logits_d2_up   = up_to_size(logits_d2,   full_size)
                logits_d3_up   = up_to_size(logits_d3,   full_size)

                loss = (aux_weights[0]*criterion_main(logits_main_up, masks) +
                        aux_weights[1]*criterion_aux (logits_d2_up,   masks) +
                        aux_weights[2]*criterion_aux (logits_d3_up,   masks))
                val_loss += loss.item()
                miou, f1 = mIoU(logits_main_up, masks, n_classes=n_classes)
                val_iou += miou
                val_f1  += f1
                val_acc += pixel_accuracy(logits_main_up, masks)

        mean_val_loss = val_loss/len(val_loader)
        mean_val_iou  = val_iou/len(val_loader)
        mean_val_f1   = val_f1/len(val_loader)
        mean_val_acc  = val_acc/len(val_loader)

        history['train_loss'].append(mean_train_loss); history['val_loss'].append(mean_val_loss)
        history['train_miou'].append(mean_train_iou);  history['val_miou'].append(mean_val_iou)
        history['train_f1'].append(mean_train_f1);     history['val_f1'].append(mean_val_f1)
        history['train_acc'].append(mean_train_acc);   history['val_acc'].append(mean_val_acc)

        print(f"Epoch:{e+1}/{epochs}.. "
              f"Train Loss: {mean_train_loss:.3f}  Val Loss: {mean_val_loss:.3f}  "
              f"Train mIoU: {mean_train_iou:.3f}  Val mIoU: {mean_val_iou:.3f}  "
              f"Train F1: {mean_train_f1:.3f}  Val F1: {mean_val_f1:.3f}  "
              f"Train Acc: {mean_train_acc:.3f}  Val Acc: {mean_val_acc:.3f}")

        # CSV
        with open(CSV_PATH, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                'epoch': e+1,
                'train_loss': mean_train_loss, 'val_loss': mean_val_loss,
                'train_miou': mean_train_iou,   'val_miou':  mean_val_iou,
                'train_f1':   mean_train_f1,    'val_f1':    mean_val_f1,
                'train_acc':  mean_train_acc,   'val_acc':   mean_val_acc
            })

        # Checkpoints
        improved = False
        if mean_val_loss < getattr(fit, 'best_loss', float('inf')):
            fit.best_loss = mean_val_loss
            torch.save(model.state_dict(), os.path.join('progress', 'SwinUNet_512_best_val_loss.pth'))
            print(f"✓ Saved best-by-loss (val_loss={mean_val_loss:.4f})"); improved = True
            not_improve = 0
        else:
            not_improve += 1

        if mean_val_iou > getattr(fit, 'best_miou', -float('inf')):
            fit.best_miou = mean_val_iou
            torch.save(model.state_dict(), os.path.join('progress', 'SwinUNet_512_best_mIoU.pth'))
            print(f"✓ Saved best-by-mIoU (val_mIoU={mean_val_iou:.4f})"); improved = True

        if not_improve >= patience:
            print(f"No val loss improvement for {patience} epochs. Stopping.")
            break

    return history

# ----------------------------
# Build data
# ----------------------------
custom_label_map = None  # set like {0:0,50:1,100:2,150:3,254:4} if needed

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

print('the number of image/label in the train: ', len(train_dataset))
print('the number of image/label in the validation: ', len(valid_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))

# ----------------------------
# Model, optimizer, scheduler
# ----------------------------
model = SwinUNet(encoder_name='swin_tiny_patch4_window7_224', num_classes=NUM_CLASSES).to(device)

# Class weights (optional)
ce_weight = None  # e.g., torch.tensor([1.0, 2.0, 2.0, 3.0, 3.0], device=device)

criterion_main = CombinedLoss(NUM_CLASSES, ce_weight=ce_weight, dice_weight=0.5, ce_weight_factor=0.5)
criterion_aux  = CombinedLoss(NUM_CLASSES, ce_weight=ce_weight, dice_weight=0.5, ce_weight_factor=0.5)

max_lr = 1e-3
epochs = 100
weight_decay = 1e-4

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
steps_per_epoch = max(1, len(train_loader))
sched = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
    pct_start=0.1, div_factor=25.0, final_div_factor=1e4
)

# ----------------------------
# Train
# ----------------------------
history = fit(
    epochs=epochs, model=model,
    train_loader=train_loader, val_loader=valid_loader,
    criterion_main=criterion_main, criterion_aux=criterion_aux,
    optimizer=optimizer, scheduler=sched,
    device=device, n_classes=NUM_CLASSES,
    aux_weights=(1.0, 0.4, 0.2), patience=20
)

print("Training complete.")



