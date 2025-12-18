# %%
"""
model_training.py
-----------------
Builds and trains the CNN model for fruit freshness classification.
Inputs: Data configuration from data_processing.py
Outputs: Trained model (.keras file) and training history
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import math
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("🧠 FRUIT FRESHNESS CLASSIFICATION - MODEL TRAINING")
print("=" * 70)

# %%
# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = Path.cwd()
DATA_CONFIG_FILE = "data_config.pkl"
MODEL_SAVE_PATH = "best_fruit_freshness_model.keras"
HISTORY_SAVE_PATH = "training_history.pkl"

# Training hyperparameters
EPOCHS = 50
LEARNING_RATE = 0.001

print(f"📁 Working directory: {BASE_PATH}")
print(f"⚙️  Epochs: {EPOCHS}")
print(f"⚙️  Learning rate: {LEARNING_RATE}")

# %%
# ============================================================================
# LOAD DATA CONFIGURATION
# ============================================================================


def load_data_config():
    """Load data configuration from data processing step."""
    if not os.path.exists(DATA_CONFIG_FILE):
        raise FileNotFoundError(
            f"❌ {DATA_CONFIG_FILE} not found! Please run 'data_processing.py' first."
        )

    with open(DATA_CONFIG_FILE, "rb") as f:
        config = pickle.load(f)

    print("\n✅ Data configuration loaded:")
    print(f"  - Image size: {config['img_size']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Number of classes: {config['num_classes']}")
    print(f"  - Training samples: {config['train_samples']}")
    print(f"  - Validation samples: {config['val_samples']}")

    return config


# %%
# ============================================================================
# RECREATE DATA GENERATORS
# ============================================================================


def recreate_generators(config):
    """Recreate data generators from saved configuration."""
    print("\n🔄 Recreating data generators...")

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=config["validation_split"],
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        config["train_dir"],
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    validation_generator = train_datagen.flow_from_directory(
        config["train_dir"],
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="categorical",
        subset="validation",
        shuffle=True,
    )

    print("✅ Generators recreated successfully")
    return train_generator, validation_generator


# %%
# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================


def create_cnn_model(input_shape=(100, 100, 3), num_classes=6):
    """Create a lightweight CNN for fruit freshness classification."""
    print("\n🧠 Building CNN Model...")

    model = keras.Sequential(
        [
            # First Convolutional Block
            layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=input_shape
            ),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


# %%
# ============================================================================
# MODEL COMPILATION
# ============================================================================


def compile_model(model, learning_rate=0.001):
    """Compile model with optimizer and metrics."""
    print("\n⚙️  Compiling model...")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    print("✅ Model compiled successfully")
    return model


# %%
# ============================================================================
# CALLBACKS SETUP
# ============================================================================


def setup_callbacks():
    """Setup training callbacks."""
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        # Learning rate reduction
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode="min",
        ),
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    print("\n📋 Callbacks configured:")
    print("  1. Early Stopping (patience=10)")
    print("  2. ReduceLROnPlateau (factor=0.5, patience=5)")
    print("  3. Model Checkpoint (save best model)")

    return callbacks


# %%
# ============================================================================
# CLASS WEIGHTS COMPUTATION
# ============================================================================


def compute_class_weights(train_generator):
    """Compute class weights for imbalanced dataset."""
    print("\n⚖️  Computing class weights...")

    y_train = train_generator.classes
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    print(f"Class weights: {class_weights_dict}")
    return class_weights_dict


# %%
# ============================================================================
# TRAINING
# ============================================================================


def train_model(model, train_gen, val_gen, callbacks, class_weights, epochs=30):
    """Train the model."""
    print("\n" + "=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)

    train_steps = math.ceil(train_gen.samples / train_gen.batch_size)
    val_steps = math.ceil(val_gen.samples / val_gen.batch_size)

    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    print("\n✅ Training completed!")
    return history


# %%
# ============================================================================
# SAVE TRAINING HISTORY
# ============================================================================


def save_training_history(history):
    """Save training history for later analysis."""
    with open(HISTORY_SAVE_PATH, "wb") as f:
        pickle.dump(history.history, f)

    print(f"\n💾 Training history saved to: {HISTORY_SAVE_PATH}")


# %%
# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        print("\n" + "=" * 70)
        print("STEP 1: LOAD DATA CONFIGURATION")
        print("=" * 70)
        config = load_data_config()

        print("\n" + "=" * 70)
        print("STEP 2: RECREATE DATA GENERATORS")
        print("=" * 70)
        train_gen, val_gen = recreate_generators(config)

        print("\n" + "=" * 70)
        print("STEP 3: BUILD MODEL")
        print("=" * 70)
        input_shape = (*config["img_size"], 3)
        model = create_cnn_model(
            input_shape=input_shape, num_classes=config["num_classes"]
        )
        model.summary()

        print("\n" + "=" * 70)
        print("STEP 4: COMPILE MODEL")
        print("=" * 70)
        model = compile_model(model, learning_rate=LEARNING_RATE)

        print("\n" + "=" * 70)
        print("STEP 5: SETUP CALLBACKS")
        print("=" * 70)
        callbacks = setup_callbacks()

        print("\n" + "=" * 70)
        print("STEP 6: COMPUTE CLASS WEIGHTS")
        print("=" * 70)
        class_weights = compute_class_weights(train_gen)

        print("\n" + "=" * 70)
        print("STEP 7: TRAIN MODEL")
        print("=" * 70)
        history = train_model(
            model, train_gen, val_gen, callbacks, class_weights, epochs=EPOCHS
        )

        print("\n" + "=" * 70)
        print("STEP 8: SAVE")
        print("=" * 70)
        save_training_history(history)

        print("\n" + "=" * 70)
        print("✅ MODEL TRAINING COMPLETE!")
        print("=" * 70)
        print(f"📊 Model saved to: {MODEL_SAVE_PATH}")
        print(f"📊 History saved to: {HISTORY_SAVE_PATH}")
        print(f"\n➡️  Next step: Run 'evaluation_testing.py'")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
