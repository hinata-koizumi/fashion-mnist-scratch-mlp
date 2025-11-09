[日本語版 (Japanese)](README_JA.md) | **English**

# University of Tokyo Deep Learning Course Competition

## Competition Results

- **Final Rank**: **3rd** / 1,439 participants
- **LB Score**: **0.9485**

## Overview

Classification of Fashion-MNIST (10 classes), the fashion version of MNIST, using a Multi-Layer Perceptron.

For details about Fashion-MNIST, please refer to the following link:
Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist

## Rules

- Training data is provided as `x_train`, `t_train`, and test data as `x_test`.
- Prediction labels should be represented as class labels 0~9, not as one-hot encoding.
- Do not use training data other than `x_train` and `t_train` specified in the cell below.
- PyTorch may be used.
- However, do not use high-level APIs such as `torch.nn.Conv2d`. Specifically, only `nn.Parameter`, `nn.Module`, and `nn.Sequential` are allowed among nn APIs. Using other APIs will result in an error.
- Do not use pre-implemented models such as those in `torchvision`.

## Approach

- Data Preprocessing/Splitting
  - Load `x_train.npy`, `t_train.npy`/`y_train.npy`, `x_test.npy` and normalize to 28×28 `float32` in [0,1]
  - Split the last 10% of training data for validation (fixed seed)
  - `DataLoader` shuffles during training, validation/test are in fixed order

- Image Augmentation (Scheduled)
  - Rotation (±8°), translation (±1.5px), scaling (±6%), shearing (±6°), horizontal flip (probability 0.25), Cutout (probability 0.5)
  - Linearly fade intensity from epoch 10→26, augmentation OFF after epoch 32

- Feature Extraction (HOG + Raw Pixels, 8296 dimensions)
  - Aggregate HOG from Sobel gradients at 3 scales: cell=4,bins=8(1152), cell=3,bins=9(2304), cell=2,bins=6(4056)
  - Concatenate raw pixels (784) and standardize using mean/variance estimated from training data

- Model (Custom MLP)
  - Fully connected layers only: 8296→3072→1536→768→10
  - Activation: GELU (custom implementation), Normalization: 1D LayerNorm (custom implementation)
  - Dropout increases during training: initial (0.10,0.12,0.15) → final (0.32,0.35,0.38)
  - Complies with `nn.*` constraints (high-level layers unused, enforced by inspection function)

- Training and Regularization
  - Optimizer: AdamW (weight_decay=6e-4)
  - Learning rate: 6-epoch warmup followed by cosine decay (base LR=2.5e-3)
  - Loss: CE with label smoothing (linearly decreases from 0.12 → 0.04)
  - Mixup: α=0.18 (active until epoch 28)
  - Gradient clipping: global norm 5.0, uses AMP + GradScaler

- EMA/SWA and Validation Selection
  - EMA: gradually strengthened from 0.9992→0.9996
  - SWA: updated in latter half of training (after augmentation stops)
  - Evaluate last/EMA/SWA each epoch and keep best weights

- TTA and Temperature Scaling / Weight Learning
  - Transform candidates: identity, ±5°/±7°, (±1,±1) translation, horizontal flip
  - This configuration adopts identity and 2 translation types (keep=[0,5,6])
  - Learn temperature T on identity logits using LBFGS, optimize class-conditional weight matrix W, and weighted combine TTA logits
  - Apply Identity anchor that boosts identity based on prediction confidence

- Prototype Correction (Optional)
  - Use class centers/variances from training features to add prototype distance + Gaussian likelihood-derived bonus to logits (only for low confidence)

- MC Dropout (Automatic)
  - Explore multiple scales on validation, enable if accuracy improvement exceeds threshold. Final prediction is average of base and MC

- Inference/Saving
  - Determine class from combined logits above and save to `data/output/submission.csv` with `label` column and `id` index

## Tech Stack

- Python 3.9+
- PyTorch (`torch.nn`, `torch.optim`, `torch.utils.data`)
- NumPy
- Pandas
- Pillow (PIL)

