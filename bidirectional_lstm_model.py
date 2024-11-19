import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import joblib
import os
import matplotlib.pyplot as plt

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
y = df['Life expectancy'].values 

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
# 2. Build the Bidirectional LSTM Model
# ---------------------------

from tensorflow.keras.layers import Bidirectional

# Build the model
model = Sequential([
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

# Define custom metric
def correlation_coefficient(y_true, y_pred):
    x = y_true - K.mean(y_true)
    y = y_pred - K.mean(y_pred)
    numerator = K.sum(x * y)
    denominator = K.sqrt(K.sum(K.square(x)) * K.sum(K.square(y)))
    return numerator / (denominator + K.epsilon())

# Compile the model
model.compile(
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
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=200,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# ---------------------------
# 5. Save the Model and Scalers
# ---------------------------

# Save the scalers
os.makedirs('models', exist_ok=True)
joblib.dump(feature_scaler, 'models/feature_scaler.save')
joblib.dump(target_scaler, 'models/target_scaler.save')

# Save the model
model.save('models/lstm_life_expectancy.h5')

# ---------------------------
# 6. Evaluate the Model
# ---------------------------

# Evaluate the model
y_pred_scaled = model.predict(X_test_seq)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test_unscaled = target_scaler.inverse_transform(y_test_seq)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test_unscaled, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))
r2 = r2_score(y_test_unscaled, y_pred)
corr_coef = np.corrcoef(y_test_unscaled.flatten(), y_pred.flatten())[0,1]

print("\nTest Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"\nPrediction Range:")
print(f"Min: {y_pred.min():.2f}, Max: {y_pred.max():.2f}")

print(f"\nCorrelation coefficient: {corr_coef:.4f}")

# ---------------------------
# 7. Visualize and Save Plots
# ---------------------------

# Create directory for figures
os.makedirs('figures', exist_ok=True)

# Plot Actual vs Predicted Life Expectancy
plt.figure(figsize=(10,6))
plt.plot(y_test_unscaled, label='Actual', marker='o')
plt.plot(y_pred, label='Predicted', marker='x')
plt.title('Actual vs Predicted Life Expectancy')
plt.xlabel('Sample Index')
plt.ylabel('Life Expectancy')
plt.legend()
plt.tight_layout()
plt.savefig('figures/actual_vs_predicted_line_plot.png')
plt.show()

# Scatter Plot of Actual vs Predicted Life Expectancy
plt.figure(figsize=(6,6))
plt.scatter(y_test_unscaled, y_pred, alpha=0.7)
plt.plot([y_test_unscaled.min(), y_test_unscaled.max()],
         [y_test_unscaled.min(), y_test_unscaled.max()], 'k--', lw=2)
plt.title('Predicted vs Actual Life Expectancy')
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.tight_layout()
plt.savefig('figures/actual_vs_predicted_scatter_plot.png')
plt.show()

# Residual Plot
residuals = y_test_unscaled.flatten() - y_pred.flatten()
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='dashed')
plt.title('Residuals Plot')
plt.xlabel('Predicted Life Expectancy')
plt.ylabel('Residuals (Actual - Predicted)')
plt.tight_layout()
plt.savefig('figures/residuals_plot.png')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(8,6))
plt.hist(residuals, bins=20, edgecolor='k')
plt.title('Distribution of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('figures/residuals_histogram.png')
plt.show()

# Plot training & validation loss
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/training_validation_loss.png')
plt.show()

# Plot training & validation MAE
plt.figure(figsize=(10,6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.tight_layout()
plt.savefig('figures/training_validation_mae.png')
plt.show()

# Plot correlation coefficient over epochs
plt.figure(figsize=(10,6))
plt.plot(history.history['correlation_coefficient'], label='Training Correlation Coefficient')
plt.plot(history.history['val_correlation_coefficient'], label='Validation Correlation Coefficient')
plt.title('Correlation Coefficient over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Correlation Coefficient')
plt.legend()
plt.tight_layout()
plt.savefig('figures/correlation_coefficient_over_epochs.png')
plt.show()

# ---------------------------
# 8. Additional Visualizations
# ---------------------------

# Feature Importance using Permutation Importance (Optional)
from sklearn.inspection import permutation_importance

# Since LSTM models are not directly compatible with permutation importance,
# we'll use a simple approximation by training a Random Forest as a surrogate model.

from sklearn.ensemble import RandomForestRegressor

# Train a Random Forest on the scaled training data
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train_scaled.ravel())

# Compute permutation importance on the test data
perm_importance = permutation_importance(rf, X_test_scaled, y_test_scaled.ravel(), n_repeats=10, random_state=42)

# Get feature importances
feature_importances = perm_importance.importances_mean

# Plot Feature Importances
sorted_idx = feature_importances.argsort()
plt.figure(figsize=(10,8))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(feature_columns)[sorted_idx])
plt.title('Feature Importances from Random Forest')
plt.xlabel('Mean Decrease in Impurity')
plt.tight_layout()
plt.savefig('figures/feature_importances.png')
plt.show()

# Save the feature importances to a CSV file
feature_importance_df = pd.DataFrame({
    'Feature': np.array(feature_columns)[sorted_idx],
    'Importance': feature_importances[sorted_idx]
})
feature_importance_df.to_csv('figures/feature_importances.csv', index=False)
