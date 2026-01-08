import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load all data
data0 = np.load('./data0.npy')
data1 = np.load('./data1.npy')
data2 = np.load('./data2.npy')
lab0 = np.load('./lab0.npy')
lab1 = np.load('./lab1.npy')
lab2 = np.load('./lab2.npy')

# Combine all data
X = np.concatenate([data0, data1, data2])
y = np.concatenate([lab0, lab1, lab2])

# Normalize pixel values
X = X.astype('float32') / 255.0

# Ensure correct shape for CNN
if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)

# Use the same train/test split (80/20)
split_idx = int(0.8 * len(X))
X_test = X[split_idx:]
y_test = y[split_idx:]

print(f"Test set size: {X_test.shape}")
print(f"Number of test samples: {len(X_test)}")
print()

# Load both models
print("Loading models...")
baseline_model = keras.models.load_model('digit_sum_cnn_baseline.keras')
improved_model = keras.models.load_model('best_model.keras')
print("Models loaded successfully!")
print()

# Evaluate baseline model
print("="*70)
print("BASELINE MODEL EVALUATION")
print("="*70)

baseline_predictions = baseline_model.predict(X_test, verbose=0)
baseline_correct = 0

for i, (pred, true_val) in enumerate(zip(baseline_predictions, y_test)):
    pred_rounded = round(pred[0])
    true_rounded = round(true_val)
    is_correct = (pred_rounded == true_rounded)
    
    if is_correct:
        baseline_correct += 1
    
    print(f"Image {i+1:4d}: Predicted = {pred_rounded:2d}, True = {true_rounded:2d} {'✓' if is_correct else '✗'}")

baseline_accuracy = (baseline_correct / len(y_test)) * 100
print()
print(f"Baseline Model Accuracy: {baseline_correct}/{len(y_test)} = {baseline_accuracy:.2f}%")
print()

# Evaluate improved model
print("="*70)
print("IMPROVED MODEL EVALUATION")
print("="*70)

improved_predictions = improved_model.predict(X_test, verbose=0)
improved_correct = 0

for i, (pred, true_val) in enumerate(zip(improved_predictions, y_test)):
    pred_rounded = round(pred[0])
    true_rounded = round(true_val)
    is_correct = (pred_rounded == true_rounded)
    
    if is_correct:
        improved_correct += 1
    
    print(f"Image {i+1:4d}: Predicted = {pred_rounded:2d}, True = {true_rounded:2d} {'✓' if is_correct else '✗'}")

improved_accuracy = (improved_correct / len(y_test)) * 100
print()
print(f"Improved Model Accuracy: {improved_correct}/{len(y_test)} = {improved_accuracy:.2f}%")
print()

# Summary comparison
print("="*70)
print("SUMMARY")
print("="*70)
print(f"Baseline Model: {baseline_accuracy:.2f}% ({baseline_correct}/{len(y_test)} correct)")
print(f"Improved Model: {improved_accuracy:.2f}% ({improved_correct}/{len(y_test)} correct)")
print(f"Improvement: {improved_accuracy - baseline_accuracy:+.2f}%")
print()
