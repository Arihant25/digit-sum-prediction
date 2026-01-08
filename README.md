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

1. Residual connections - 2 residual blocks with skip connections for better gradient flow
2. Batch normalization - After each convolution for faster convergence
3. Minimal data augmentation - Light rotation (±5°) and translation (±5%) to avoid distorting digits
4. MSE loss - Consistent with baseline for fair comparison
5. Adam optimizer - Standard Adam with learning rate decay via ReduceLROnPlateau
6. Progressive dropout - (0.25 → 0.3 → 0.4) through the network
7. Flatten layer - Preserves more spatial information than GlobalAveragePooling

### Model Architecture

- Minimal data augmentation layer (rotation ±5°, translation ±5%)
- Initial conv (32 filters) + BatchNorm + ReLU
- 2 residual blocks (64, 128 filters) with skip connections
- MaxPooling and progressive dropout (0.25, 0.3) after each block
- Flatten layer (preserves spatial info)
- Dense layer (128 units) with dropout (0.4)
- Output layer (1 unit for regression)
- **Total params**: ~600K

### Training Details

- **Optimizer**: Adam (learning rate=0.001)
- **Loss**: MSE
- **Epochs**: Up to 40 (early stopping patience=10)
- **Batch Size**: 64
- **Validation**: 20% of training data (implicit split via `validation_split=0.2`)
- **Final Split**: ~64% training, ~16% validation, ~20% test
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Usage

```bash
python improved_model.py
```

Model saves as `best_model.keras`

### Comparison

| Model    | Test MAE | Test Accuracy | Training Time (CPU) | Architecture                      |
|----------|----------|---------------|---------------------|-----------------------------------|
| Baseline | ~1.93    | 16.83%        | ~30 mins            | Simple 3-layer CNN                |
| Improved | ~7.53    | 3.68%         | ~10 hours           | ResNet-inspired with residuals    |

**Note**: MAE (Mean Absolute Error) is used as the primary metric since this is a regression task (predicting continuous digit sums). Using "accuracy" (exact match count) would be too strict for continuous predictions.

## Model Evaluation

Compare both models on the test set:

```bash
python evaluate.py
```

This script:
- Uses the same 80/20 train/test split as the training scripts
- Loads both `digit_sum_cnn_baseline.keras` and `best_model.keras`
- Runs inference on all test samples
- Calculates accuracy by rounding predictions and comparing to true values
- Prints live predictions for each image (predicted vs true value)
- Displays final accuracy percentage for both models

**Accuracy Metric**: Predictions are rounded to nearest integer and compared with rounded true values. A correct prediction means `round(predicted) == round(true_value)`.

## Data Visualization

View random samples from each dataset:

```bash
uv run visualize_samples.py
```

This script:
- Loads one random image from each of data0, data1, and data2
- Displays the corresponding labels
- Saves visualized images to the `smell/` folder as PNG files

## Files

- `baseline.py` - Baseline CNN
- `improved_model.py` - Improved model with residual blocks
- `evaluate.py` - Evaluation script comparing both models on test set
- `visualize_samples.py` - Visualization script for random dataset samples
- `data*.npy`, `lab*.npy` - Training data
- `*.keras` - Saved models
- `smell/` - Output folder for visualized samples
