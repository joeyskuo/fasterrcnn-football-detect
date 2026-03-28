# Faster R-CNN Football & Player Detection

Object detection model that identifies footballs and players in images using a fine-tuned Faster R-CNN (ResNet-50 FPN backbone) from torchvision.

## Skills Demonstrated

- PyTorch model training pipeline (data loading → training loop → validation → checkpointing)
- Transfer learning: fine-tuning a pretrained Faster R-CNN detection head for a custom class set
- Custom `torch.utils.data.Dataset` implementation with YOLO-format label parsing
- mAP evaluation using `torchmetrics.detection.MeanAveragePrecision`
- Google Colab + Google Drive workflow for cloud GPU training

## Model

- **Architecture:** Faster R-CNN, ResNet-50 FPN backbone
- **Pretrained weights:** COCO (torchvision `DEFAULT`)
- **Classes:** background, football, player (`NUM_CLASSES = 3`)
- **Head:** replaced `FastRCNNPredictor` to match class count

## Dataset

YOLO format — images in `images/`, labels in `labels/` as `.txt` files with normalized `class cx cy bw bh` per line. Split into `train/` and `valid/` directories.

- Class 0 → football (mapped to label index 1)
- Class 1 → player (mapped to label index 2)

## Training

Configured in `fasterrcnn_training.ipynb`:

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD |
| Learning rate | 0.005 |
| Momentum | 0.9 |
| Weight decay | 0.005 |
| LR scheduler | StepLR (step=5, gamma=0.5) |
| Epochs | 20 |
| Batch size | 4 |

Best checkpoint saved to `best.pth` based on highest `mAP@0.5` on the validation set.

## Validation Metrics

Computed each epoch using `torchmetrics`:
- `mAP@0.5` — IoU threshold 0.5
- `mAP@0.5:0.95` — COCO standard, averaged across IoU thresholds 0.5–0.95

## Requirements

```
torch
torchvision
torchmetrics
Pillow
```

## Usage

Open `fasterrcnn_training.ipynb` in Google Colab. Mount your Drive, update `DATA_ROOT` to point to your dataset, and run all cells.
