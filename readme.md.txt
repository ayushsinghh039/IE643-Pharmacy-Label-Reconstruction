# Lightweight Diffusion-Based Recovery of Missing Pharmacy Label Information

## IE643 Deep Learning Course Project

**Team:** Maverick

## Project Overview

Medication labels may be partially obscured, blurred, cropped, or damaged during automated pharmacy scanning or delivery.

The objective of this project is to reconstruct only the unreadable region of a pharmacy label while preserving the visible information.

The overall pipeline is:

Damaged Image
↓
Missing Region Detection
↓
Predicted Damage Mask
↓
Lightweight Diffusion Reconstruction
↓
Recovered Pharmacy Label

## Important Input Design

The model receives only the damaged image as input.

The ground-truth damage mask is used during training and evaluation, but is not provided to the model during inference.

## Dataset

Total images: 6794

Train: 4755  
Validation: 1019  
Test: 1020

Input resolution:

256 × 256

Damage types include cut, blur, paint/occlusion, and combined damage.

## Missing Region Detection

We are comparing:

1. U-Net
2. DeepLabV3
3. YOLO Segmentation

The same train/validation/test split is used for all models.

## U-Net Baseline

The first model implemented is U-Net.

### Test Results

| Metric | Result |
|---|---:|
| Test Dice | 0.9477 |
| Test IoU | 0.9072 |
| Test Loss | 0.0888 |

### Qualitative Results

![U-Net Mask Results](results/unet/unet_mask_results.png)

## Project Status

- [x] Dataset preparation
- [x] Train/validation/test split
- [x] U-Net damage mask detection
- [x] U-Net test evaluation
- [x] U-Net qualitative evaluation
- [ ] DeepLabV3
- [ ] YOLO segmentation
- [ ] Model comparison
- [ ] Lightweight diffusion reconstruction
- [ ] End-to-end inference pipeline
- [ ] User interface

## Team

Team Maverick

IE643 Deep Learning Project