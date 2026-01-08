import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, using CPU")


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

# Normalization
X = X.astype('float32') / 255.0

# Ensure correct shape for CNN
if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)

# Train/test split (80/20)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Minimal data augmentation
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05)
])

# Improved CNN with residual connections
def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

inputs = layers.Input(shape=X.shape[1:])
x = data_augmentation(inputs)

# Initial conv
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)

# 2 residual blocks
x = residual_block(x, 64)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)

x = residual_block(x, 128)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)

# Flatten to preserve spatial info
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1)(x)

model = keras.Model(inputs, x)

# Use Adam optimizer with MSE loss
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss='mse', 
              metrics=['mae'])
model.summary()

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_mae', patience=10, restore_best_weights=True, mode='min'
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_mae', factor=0.5, patience=5, min_lr=1e-6, mode='min'
)
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras', monitor='val_mae', save_best_only=True, mode='min'
)

start_time = time.time()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

end_time = time.time()
training_time = end_time - start_time

print(f"Training time: {training_time:.2f} seconds")

# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {test_mae:.2f}")

# Best model already saved via checkpoint callback
print("Best model saved as best_model.keras")
