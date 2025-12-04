import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load CSV file
df = pd.read_csv("Data/iris.csv")

X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]


# Encode text labels (setosa, versicolor, virginica → 0,1,2)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# One-hot encode target
y_onehot = tf.keras.utils.to_categorical(y_encoded)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_onehot, test_size=0.2, random_state=42
)

model = keras.Sequential(
    [
        keras.Input(shape=[4]),
        layers.Dense(8, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
# For **multi-class classification** → use **Categorical Cross-Entropy** (with softmax)

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=2,
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.3f}")

# Predict
predictions = model.predict(X_test)
print(predictions[:5])
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)

print("\nPredicted Classes:", predicted_classes[:10])
print("Actual Classes:   ", actual_classes[:10])

