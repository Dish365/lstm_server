# bidirectional_lstm_training.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------

# Load data
df = pd.read_csv('Data_GHGE.csv')

# Feature selection
feature_columns = [
    'FP index ', 'LP index ', 'Vegetal Pds-FS', 'Cereals -FS', 'Starchy Rts-FS',
    'Pulses-FS', 'Fruits -FS', 'Meat-FS', 'Fish-FS', 'Sugar & Swt-FS',
    'Oils-FS ', 'Vegetables-FS ', 'Spices-FS', 'Eggs-FS', 'Milk-FS',
    'Cereals-LSF', 'Starchy Rts-LSF', 'Pulses-LSF', 'Meat-LSF', 'Fish-LSF',
    'Cereals-LS', 'Starchy-LS', 'Fruits-LS', 'Energy use', 'Renewable energy '
]

# Ensure feature names match exactly
feature_columns = [col.strip() for col in feature_columns]
df.columns = [col.strip() for col in df.columns]

X = df[feature_columns].values
y = df['Life expectancy'].values  # Ensure the target column name matches exactly

# Split data before scaling to prevent data leakage
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.15, random_state=42,
    stratify=pd.qcut(y, q=5, labels=False)
)

# Initialize scalers
feature_scaler = RobustScaler()
target_scaler = RobustScaler()

# Fit scalers on training data
X_train_scaled = feature_scaler.fit_transform(X_train_raw)
y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1))

# Transform test data
X_test_scaled = feature_scaler.transform(X_test_raw)
y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1))

# Create sequences
def create_sequences(X, y, sequence_length=3):
    sequences_X = []
    sequences_y = []
    for i in range(len(X) - sequence_length + 1):
        sequences_X.append(X[i:i+sequence_length])
        sequences_y.append(y[i+sequence_length-1])
    return np.array(sequences_X), np.array(sequences_y)

# Create sequences in training data
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length=3)

# Create sequences in test data
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length=3)

# ---------------------------
# 2. Build Bidirectional LSTM Model
# ---------------------------

from tensorflow.keras.regularizers import l2

# Build the Bidirectional LSTM model
model_bi = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)),
                  input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(0.3),
    
    Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))),
    Dropout(0.3),
    
    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='linear')
])

# ---------------------------
# 3. Compile the Model
# ---------------------------

# Define custom metric for correlation coefficient
def correlation_coefficient(y_true, y_pred):
    x = y_true - K.mean(y_true)
    y = y_pred - K.mean(y_pred)
    numerator = K.sum(x * y)
    denominator = K.sqrt(K.sum(K.square(x)) * K.sum(K.square(y)))
    return numerator / (denominator + K.epsilon())

# Compile the model
model_bi.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae', correlation_coefficient]
)

# ---------------------------
# 4. Train the Model
# ---------------------------

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_correlation_coefficient', mode='max', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)
]

# Train the model
history_bi = model_bi.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=1000,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# ---------------------------
# 5. Evaluate the Model
# ---------------------------

# Predict on test data
y_pred_bi_scaled = model_bi.predict(X_test_seq)
y_pred_bi = target_scaler.inverse_transform(y_pred_bi_scaled)
y_test_unscaled = target_scaler.inverse_transform(y_test_seq)

# Calculate evaluation metrics
mae_bi = mean_absolute_error(y_test_unscaled, y_pred_bi)
rmse_bi = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_bi))
r2_bi = r2_score(y_test_unscaled, y_pred_bi)
corr_bi = np.corrcoef(y_test_unscaled.flatten(), y_pred_bi.flatten())[0,1]

print("\nBidirectional LSTM Model Test Metrics:")
print(f"MAE: {mae_bi:.4f}")
print(f"RMSE: {rmse_bi:.4f}")
print(f"R2: {r2_bi:.4f}")
print(f"Correlation Coefficient: {corr_bi:.4f}")

# ---------------------------
# 6. Compare with Unidirectional LSTM Model
# ---------------------------

# Build the Unidirectional LSTM model (for comparison)
model_uni = Sequential([
    LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=True, 
         kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    
    LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    
    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1, activation='linear')
])

# Compile the model
model_uni.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae', correlation_coefficient]
)

# Train the Unidirectional LSTM model
history_uni = model_uni.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=1000,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# Predict on test data
y_pred_uni_scaled = model_uni.predict(X_test_seq)
y_pred_uni = target_scaler.inverse_transform(y_pred_uni_scaled)

# Calculate evaluation metrics for the unidirectional LSTM model
mae_uni = mean_absolute_error(y_test_unscaled, y_pred_uni)
rmse_uni = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_uni))
r2_uni = r2_score(y_test_unscaled, y_pred_uni)
corr_uni = np.corrcoef(y_test_unscaled.flatten(), y_pred_uni.flatten())[0,1]

print("\nUnidirectional LSTM Model Test Metrics:")
print(f"MAE: {mae_uni:.4f}")
print(f"RMSE: {rmse_uni:.4f}")
print(f"R2: {r2_uni:.4f}")
print(f"Correlation Coefficient: {corr_uni:.4f}")

# ---------------------------
# 7. Compare the Models
# ---------------------------

# Create a DataFrame to display results
import pandas as pd

results = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R2 Score', 'Correlation Coefficient'],
    'Unidirectional LSTM': [mae_uni, rmse_uni, r2_uni, corr_uni],
    'Bidirectional LSTM': [mae_bi, rmse_bi, r2_bi, corr_bi]
})

print("\nComparison of Model Performance:")
print(results)

# ---------------------------
# 8. Visualize Predictions
# ---------------------------

# Plot Actual vs Predicted for Unidirectional LSTM
plt.figure(figsize=(12,6))
plt.plot(y_test_unscaled, label='Actual')
plt.plot(y_pred_uni, label='Unidirectional LSTM Prediction')
plt.title('Unidirectional LSTM - Actual vs Predicted Life Expectancy')
plt.xlabel('Sample Index')
plt.ylabel('Life Expectancy')
plt.legend()
plt.show()

# Plot Actual vs Predicted for Bidirectional LSTM
plt.figure(figsize=(12,6))
plt.plot(y_test_unscaled, label='Actual')
plt.plot(y_pred_bi, label='Bidirectional LSTM Prediction')
plt.title('Bidirectional LSTM - Actual vs Predicted Life Expectancy')
plt.xlabel('Sample Index')
plt.ylabel('Life Expectancy')
plt.legend()
plt.show()

# Plot Residuals for Unidirectional LSTM
residuals_uni = y_test_unscaled.flatten() - y_pred_uni.flatten()
plt.figure(figsize=(8,6))
plt.scatter(y_pred_uni, residuals_uni, alpha=0.5)
plt.title('Unidirectional LSTM - Residuals Plot')
plt.xlabel('Predicted Life Expectancy')
plt.ylabel('Residuals')
plt.hlines(y=0, xmin=y_pred_uni.min(), xmax=y_pred_uni.max(), colors='r')
plt.show()

# Plot Residuals for Bidirectional LSTM
residuals_bi = y_test_unscaled.flatten() - y_pred_bi.flatten()
plt.figure(figsize=(8,6))
plt.scatter(y_pred_bi, residuals_bi, alpha=0.5)
plt.title('Bidirectional LSTM - Residuals Plot')
plt.xlabel('Predicted Life Expectancy')
plt.ylabel('Residuals')
plt.hlines(y=0, xmin=y_pred_bi.min(), xmax=y_pred_bi.max(), colors='r')
plt.show()

# ---------------------------
# 9. Additional Analysis (Optional)
# ---------------------------

# Plot training & validation loss for Bidirectional LSTM
plt.figure(figsize=(10,6))
plt.plot(history_bi.history['loss'], label='Training Loss')
plt.plot(history_bi.history['val_loss'], label='Validation Loss')
plt.title('Bidirectional LSTM - Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot correlation coefficient over epochs for Bidirectional LSTM
plt.figure(figsize=(10,6))
plt.plot(history_bi.history['correlation_coefficient'], label='Training Correlation')
plt.plot(history_bi.history['val_correlation_coefficient'], label='Validation Correlation')
plt.title('Bidirectional LSTM - Correlation Coefficient over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Correlation Coefficient')
plt.legend()
plt.show()

# ---------------------------
# End of Script
# ---------------------------
