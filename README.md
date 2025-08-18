# Swin-UNet Cardiac MRI Segmentation

This project implements **Swin-UNet** in PyTorch for semantic segmentation of cardiac MRI images.  
It supports training, validation, and inference with visualization.

---

## Dataset Structure

dataset/
├── train/ # Training images
├── trainannot/ # Training masks
├── val/ # Validation images
└── valannot/ # Validation masks


## Requirements

Install dependencies:

```bash
pip install torch torchvision timm albumentations opencv-python matplotlib tqdm
```

## Training

Run training with:

```bash
python main_version2.py
```

## Inference & Visualization

Use the provided script to visualize predictions:

```bash
from inference_vis import show_dir

show_dir(
    images_dir="dataset/val",
    masks_dir="dataset/valannot",
    weights="progress/SwinUNet_512_best_val_loss.pth",
    max_items=8
)
```