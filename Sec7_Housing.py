import tensorflow as tf
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
cali = pd.read_csv("Data/housing.csv")
df = cali.copy()

df = df.drop("ocean_proximity", axis=1)

# Handle missing values if any
df = df.dropna()

scaler = StandardScaler()
scaled = scaler.fit_transform(df)
df = pd.DataFrame(scaled, columns=df.columns)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Improved Model Architecture
model = keras.Sequential(
    [
        keras.Input(shape=[X_train.shape[1]]),
        layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
        layers.Dense(1),  # Linear activation for regression
    ]
)

# Improved training configuration
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",  # Try MSE first, it provides stronger gradients
    metrics=["mae", "mse"],
)

# Add early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

# Train with more epochs and monitoring
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1,
)

# Evaluation
test_loss, test_mae, test_mse = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.3f}, Test MAE: {test_mae:.3f}")

# Predictions
predictions = model.predict(X_test)
preds = predictions.flatten()
actual = np.array(y_test)

print("\nFirst 10 predictions vs actual:")
for p, a in zip(preds[:10], actual[:10]):
    print(f"Predicted: {p:.3f}   Actual: {a:.3f}   Diff: {abs(p-a):.3f}")
