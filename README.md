# Digit Sum Prediction

Two ML models to predict the sum of digits in an image.

This is the assignment for the course Machine Learning for Natural Sciences (2026) at IIIT Hyderabad.

## Baseline CNN

### Model Architecture

- Standard CNN with 3 convolutional layers (32, 64, 64 filters)
- MaxPooling after each conv layer
- Fully connected layer with 64 units
- Dropout (0.5) for regularization
- Single output neuron (regression)

### Training Details

- **Data**: 30,000 images (data0-2.npy), combined and split 80/20 train/test
- **Validation**: 20% of training data (implicit split via `validation_split=0.2`)
- **Loss**: MSE (regression task)
- **Optimizer**: Adam
- **Epochs**: 20
- **Batch Size**: 32

### Usage

```bash
python baseline.py
```

Model saves as `digit_sum_cnn_baseline.keras`

### Results

- **Test MAE**: ~1.65

## Improved Model

### Key Improvements

1. Efficient architecture - 3 residual blocks with skip connections
2. Enhanced preprocessing - Per-image contrast normalization instead of simple scaling
3. Moderate data augmentation - Rotation and translation
4. Huber loss instead of MSE (more robust to outliers)
5. AdamW optimizer with weight decay and cosine learning rate decay
6. L2 regularization on dense layers
7. Progressive dropout (0.2 → 0.4) through the network

### Model Architecture

- Data augmentation layer (rotation ±10°, translation ±10%)
- Initial conv (32 filters) + BatchNorm
- 3 residual blocks (64, 128, 128 filters)
- MaxPooling and progressive dropout after each block
- Global average pooling
- Dense layers (128 → 64 → 1) with L2 regularization
- Dropout (0.4, 0.3)
- **Total params**: ~400K

### Training Details

- **Optimizer**: AdamW with cosine decay (1e-3 → 1e-5) + weight decay
- **Loss**: Huber (robust to outliers)
- **Epochs**: Up to 30 (early stopping patience=8)
- **Batch Size**: 128
- **Validation**: 20% of training data (implicit split via `validation_split=0.2`)
- **Final Split**: ~64% training, ~16% validation, ~20% test
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Usage

```bash
python improved_model.py
```

Model saves as `digit_sum_improved.keras`

### Comparison

| Model    | Test MAE | Architecture                   |
|----------|----------|--------------------------------|
| Baseline | ~1.65    | Simple 3-layer CNN             |
| Improved | ~1.64    | ResNet-inspired + augmentation |

**Note**: MAE (Mean Absolute Error) is used as the primary metric since this is a regression task (predicting continuous digit sums). Using "accuracy" (exact match count) would be too strict for continuous predictions.

## Files

- `baseline.py` - Baseline CNN
- `improved_model.py` - Improved model with residual blocks
- `data*.npy`, `lab*.npy` - Training data
- `*.keras` - Saved models
