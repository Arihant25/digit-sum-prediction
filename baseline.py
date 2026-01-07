import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# Load all training data
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

# Train/test split (80/20)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Build CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

start_time = time.time()

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

end_time = time.time()
training_time = end_time - start_time

# Evaluate on test set
print(f"Training time: {training_time:.2f} seconds")

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {test_mae:.2f}")

# Save model
model.save('digit_sum_cnn_baseline.keras')
print("Model saved as 'digit_sum_cnn_baseline.keras'")