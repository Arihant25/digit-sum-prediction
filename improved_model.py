import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

# Enable mixed precision for faster training on GPU
tf.keras.mixed_precision.set_global_policy('mixed_float16')

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

# Enhanced preprocessing: contrast normalization
X = X.astype('float32')
X = (X - X.mean(axis=(1,2), keepdims=True)) / (X.std(axis=(1,2), keepdims=True) + 1e-7)

# Ensure correct shape for CNN
if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)

# Train/test split (80/20)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

# Deeper CNN with residual connections and attention
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
x = layers.Conv2D(32, (3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

# 3 residual blocks with filters
x = residual_block(x, 64)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.2)(x)

x = residual_block(x, 128)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)

x = residual_block(x, 128)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)

# Dense layers
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(1, dtype='float32')(x)  # Output layer in float32 for numerical stability

model = keras.Model(inputs, x)

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=2000,
    alpha=1e-5
)
optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
model.summary()

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_mae', patience=8, restore_best_weights=True, mode='min'
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_mae', factor=0.5, patience=7, min_lr=1e-7, mode='min'
)
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras', monitor='val_mae', save_best_only=True, mode='min'
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Evaluate on test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {test_mae:.2f}")

# Save model
model.save('digit_sum_improved.keras')
print("Model saved as 'digit_sum_improved.keras'")
