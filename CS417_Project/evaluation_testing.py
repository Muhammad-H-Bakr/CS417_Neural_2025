"""
evaluation_testing.py
---------------------
Comprehensive evaluation and testing of the trained fruit freshness model.
Inputs: Trained model and data configuration
Outputs: Metrics, confusion matrix, classification report, and visualizations
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


print("FRUIT FRESHNESS CLASSIFICATION - EVALUATION & TESTING")


# CONFIGURATION

BASE_PATH = Path.cwd()
DATA_CONFIG_FILE = "data_config.pkl"
MODEL_PATH = "best_fruit_freshness_model.keras"
HISTORY_PATH = "training_history.pkl"

print(f" Working directory: {BASE_PATH}")

# LOAD CONFIGURATION & MODEL


def load_data_config():
    """Load data configuration."""
    if not os.path.exists(DATA_CONFIG_FILE):
        raise FileNotFoundError(
            f" {DATA_CONFIG_FILE} not found! Please run 'data_processing.py' first."
        )

    with open(DATA_CONFIG_FILE, "rb") as f:
        config = pickle.load(f)

    print("\n Data configuration loaded")
    return config


def load_trained_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f" {MODEL_PATH} not found! Please run 'model_training.py' first."
        )

    model = keras.models.load_model(MODEL_PATH)
    print(f"\n Model loaded from: {MODEL_PATH}")
    return model


def load_training_history():
    """Load training history if available."""
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            history = pickle.load(f)
        print(f" Training history loaded from: {HISTORY_PATH}")
        return history
    else:
        print("  Training history not found")
        return None


# RECREATE TEST GENERATOR


def recreate_test_generator(config):
    """Recreate test data generator."""
    print("\n Recreating test generator...")

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        config["test_dir"],
        target_size=config["img_size"],
        batch_size=config["batch_size"],
        class_mode="categorical",
        shuffle=False,
    )

    print(f" Test generator created: {test_generator.samples} samples")
    return test_generator


# MODEL EVALUATION
def evaluate_model(model, test_generator):
    """Comprehensive evaluation of model on test set."""
    print("\n EVALUATING MODEL ON TEST SET")

    test_generator.reset()
    results = model.evaluate(test_generator, verbose=1)


    loss = results[0]
    accuracy = results[1]

    # Handle cases where model might have different number of metrics
    if len(results) >= 5:
        precision = results[2]
        recall = results[3]
        auc = results[4]
    elif len(results) >= 3:
        precision = results[2]
        recall = results[2]  # Same as precision if only 3 metrics
        auc = results[2]
    else:
        precision = accuracy
        recall = accuracy
        auc = accuracy


    print(" TEST SET RESULTS")

    print(f"  Loss:      {loss:.4f}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  AUC:       {auc:.4f}")

    # Calculate F1-Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f"  F1-Score:  {f1_score:.4f}")
    else:
        f1_score = 0.0
        print(f"  F1-Score:  N/A")

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "f1_score": f1_score,
    }


# DETAILED CLASSIFICATION REPORT


def generate_classification_report(model, test_generator):
    """Generate detailed classification report and confusion matrix."""

    print(" DETAILED CLASSIFICATION REPORT")


    test_generator.reset()

    # Get predictions
    print("\n Generating predictions...")
    y_pred_proba = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_generator.classes

    # Get class names
    class_names = list(test_generator.class_indices.keys())

    # Classification report

    print(" CLASSIFICATION REPORT")

    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 12},
    )
    plt.title(
        "Confusion Matrix - Fruit Freshness Classification", fontsize=16, weight="bold"
    )
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("\n Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()

    # Per-class accuracy

    print(" PER-CLASS ACCURACY")
    for i, class_name in enumerate(class_names):
        class_correct = cm[i, i]
        class_total = np.sum(cm[i, :])
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(
            f"  {class_name:20s}: {class_correct:4d}/{class_total:4d} = {class_accuracy:.4f} ({class_accuracy*100:6.2f}%)"
        )

    return y_true, y_pred, y_pred_proba, class_names, cm


# ============================================================================
# VISUALIZE PREDICTIONS
# ============================================================================


def visualize_predictions(model, test_generator, config, num_per_class=6):
    """Visualize balanced sample predictions from each class."""
    print("\n" + "=" * 70)
    print(f"🖼️  VISUALIZING PREDICTIONS ({num_per_class} samples per class)")
    print("=" * 70)

    class_names = list(test_generator.class_indices.keys())
    num_classes = len(class_names)
    img_size = config["img_size"]
    test_dir = config["test_dir"]

    # Create figure with appropriate dimensions
    if num_classes == 1:
        fig, axes = plt.subplots(
            1,
            min(num_per_class, len(os.listdir(os.path.join(test_dir, class_names[0])))),
            figsize=(3 * num_per_class, 3),
        )
        axes = np.array([axes]) if num_per_class > 1 else np.array([[axes]])
    else:
        fig, axes = plt.subplots(
            num_classes, num_per_class, figsize=(3 * num_per_class, 3 * num_classes)
        )
        # Ensure axes is always 2D
        if num_per_class == 1:
            axes = axes.reshape(-1, 1)

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        all_images = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if len(all_images) == 0:
            continue

        selected_images = np.random.choice(
            all_images, size=min(num_per_class, len(all_images)), replace=False
        )

        for sample_idx, img_name in enumerate(selected_images):
            img_path = os.path.join(class_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            img_array_expanded = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array_expanded, verbose=0)
            pred_class_idx = np.argmax(prediction)
            pred_class = class_names[pred_class_idx]
            confidence = np.max(prediction)

            is_correct = class_idx == pred_class_idx
            color = "green" if is_correct else "red"

            ax = (
                axes[class_idx, sample_idx]
                if num_per_class > 1
                else axes[class_idx][sample_idx]
            )
            ax.imshow(img_array)
            short_name = img_name[:15] + "..." if len(img_name) > 18 else img_name
            ax.set_title(
                f"True: {class_name}\nPred: {pred_class}\nConf: {confidence:.1%}",
                color=color,
                fontsize=9,
                weight="bold",
            )
            ax.axis("off")

    plt.suptitle(
        f"Prediction Visualization: {num_per_class} Samples per Class\n(Green=Correct, Red=Incorrect)",
        fontsize=14,
        weight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig("prediction_visualization.png", dpi=300, bbox_inches="tight")
    print("✅ Prediction visualization saved as 'prediction_visualization.png'")
    plt.show()

    # Summary
    print(f"\n{'='*70}")
    print("📊 CLASS SUMMARY")
    print(f"{'='*70}")
    for class_name in class_names:
        class_path = os.path.join(test_dir, class_name)
        total = len(
            [
                f
                for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
        )
        print(f"  {class_name:20s}: {total:4d} total images")


# ============================================================================
# PLOT TRAINING HISTORY (IF AVAILABLE)
# ============================================================================


def plot_training_history(history):
    """Plot training history if available."""
    if history is None:
        print("\n⚠️  Training history not available for plotting")
        return

    print("\n📊 Plotting training history...")

    # Determine which metrics are available in history
    available_metrics = [key for key in history.keys() if not key.startswith("val_")]

    # Create appropriate subplot grid
    num_metrics = len(available_metrics)
    if num_metrics <= 3:
        nrows = 1
        ncols = num_metrics
    else:
        nrows = 2
        ncols = (num_metrics + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

    # Handle single subplot case
    if num_metrics == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for idx, metric in enumerate(available_metrics):
        if idx >= len(axes):
            break

        metric_name = metric.replace("_", " ").title()
        axes[idx].plot(history[metric], label=f"Training {metric_name}", linewidth=2)

        val_metric = f"val_{metric}"
        if val_metric in history:
            axes[idx].plot(
                history[val_metric], label=f"Validation {metric_name}", linewidth=2
            )

        axes[idx].set_title(f"{metric_name} over Epochs", fontsize=12, weight="bold")
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel(metric_name)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Training History", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig("training_history_eval.png", dpi=300, bbox_inches="tight")
    print("✅ Training history plot saved as 'training_history_eval.png'")
    plt.show()


# ============================================================================
# GENERATE EVALUATION SUMMARY
# ============================================================================


def generate_evaluation_summary(metrics, cm, class_names):
    """Generate and save comprehensive evaluation summary."""
    print("\n" + "=" * 70)
    print("📝 GENERATING EVALUATION SUMMARY")
    print("=" * 70)

    summary = []
    summary.append("=" * 70)
    summary.append("FRUIT FRESHNESS CLASSIFICATION - EVALUATION SUMMARY")
    summary.append("=" * 70)
    summary.append("")
    summary.append("OVERALL METRICS:")
    summary.append(f"  Loss:      {metrics['loss']:.4f}")
    summary.append(
        f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)"
    )
    summary.append(f"  Precision: {metrics['precision']:.4f}")
    summary.append(f"  Recall:    {metrics['recall']:.4f}")
    summary.append(f"  AUC:       {metrics['auc']:.4f}")
    summary.append(f"  F1-Score:  {metrics['f1_score']:.4f}")
    summary.append("")
    summary.append("PER-CLASS ACCURACY:")

    for i, class_name in enumerate(class_names):
        class_correct = cm[i, i]
        class_total = np.sum(cm[i, :])
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        summary.append(
            f"  {class_name:20s}: {class_correct:4d}/{class_total:4d} = {class_accuracy:.4f} ({class_accuracy*100:6.2f}%)"
        )

    summary.append("")
    summary.append("=" * 70)

    summary_text = "\n".join(summary)

    with open("evaluation_summary.txt", "w") as f:
        f.write(summary_text)

    print(summary_text)
    print("\n✅ Evaluation summary saved to 'evaluation_summary.txt'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        print("\n" + "=" * 70)
        print("STEP 1: LOAD CONFIGURATION & MODEL")
        print("=" * 70)
        config = load_data_config()
        model = load_trained_model()
        history = load_training_history()

        print("\n" + "=" * 70)
        print("STEP 2: RECREATE TEST GENERATOR")
        print("=" * 70)
        test_generator = recreate_test_generator(config)

        print("\n" + "=" * 70)
        print("STEP 3: EVALUATE MODEL")
        print("=" * 70)
        metrics = evaluate_model(model, test_generator)

        print("\n" + "=" * 70)
        print("STEP 4: GENERATE CLASSIFICATION REPORT")
        print("=" * 70)
        y_true, y_pred, y_pred_proba, class_names, cm = generate_classification_report(
            model, test_generator
        )

        print("\n" + "=" * 70)
        print("STEP 5: VISUALIZE PREDICTIONS")
        print("=" * 70)
        visualize_predictions(model, test_generator, config, num_per_class=8)

        print("\n" + "=" * 70)
        print("STEP 6: PLOT TRAINING HISTORY")
        print("=" * 70)
        plot_training_history(history)

        print("\n" + "=" * 70)
        print("STEP 7: GENERATE SUMMARY")
        print("=" * 70)
        generate_evaluation_summary(metrics, cm, class_names)

        print("\n" + "=" * 70)
        print("✅ EVALUATION & TESTING COMPLETE!")
        print("=" * 70)
        print("\n📊 Generated Files:")
        print("  - confusion_matrix.png")
        print("  - prediction_visualization.png")
        print("  - training_history_eval.png")
        print("  - evaluation_summary.txt")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
