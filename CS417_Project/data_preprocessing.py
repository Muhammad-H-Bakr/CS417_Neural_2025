
"""
Simplified Data Processing Script
---------------------------------
This script prepares a unified fruit freshness dataset and builds
training/validation/test generators for model training.
"""


import os
import shutil
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import tensorflow as tf
import subprocess
import warnings
warnings.filterwarnings("ignore")


BASE_PATH = Path.cwd()
print(f"BASE_PATH: {BASE_PATH}")



# Each fruit maps to its rotten and fresh classes in the original dataset.
FRUIT_CATEGORIES = {
    "avocado": {
        "rotten": ["Avocado Black 2", "Avocado ripe 3"],
        "fresh": ["Avocado 1", "Avocado Black 1", "Avocado Green 1"],
    },
    "nut": {
        "rotten": ["Nut 1", "Nut 4", "Nut 5", "Nut Pecan 1"],
        "fresh": ["Nut 2", "Nut 3", "Nut Forest 1"],
    },
    "plum": {"rotten": ["Plum 3", "Plum 4"], "fresh": ["Plum 1", "Plum 2"]},
    "peach": {
        "rotten": ["Peach 3", "Peach 4", "Peach 5", "Peach 6"],
        "fresh": ["Peach 1", "Peach 2", "Peach Flat 1"],
    },
    "cactus": {
        "rotten": ["Cactus fruit green 1"],
        "fresh": ["Cactus fruit 1", "Cactus fruit red 1"],
    },
}



ORIGINAL_DATASET_PATH = os.path.join(BASE_PATH, 'fruits-360-100x100')
OUTPUT_PATH = os.path.join(BASE_PATH, 'multi_fruit_freshness_dataset')

TRAIN_DIR = os.path.join(OUTPUT_PATH, 'Training')
TEST_DIR = os.path.join(OUTPUT_PATH, 'Test')

DATA_CONFIG_FILE = 'data_config.pkl'
GENERATORS_INFO_FILE = 'generators_info.pkl'

IMG_SIZE = (100, 100)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


def ensure_dataset_exists():
    """
    Ensures the original dataset exists.
    If missing clone from git
    """
    if Path(ORIGINAL_DATASET_PATH).exists():
        print(f"Dataset found at: {ORIGINAL_DATASET_PATH}")
        return True
    else:
        print(f" Dataset not found at: {ORIGINAL_DATASET_PATH}")
        print(" Cloning fruits-360 dataset...")

        try:
            # Use subprocess to clone the repository
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/fruits-360/fruits-360-100x100.git",
                ],
                cwd=BASE_PATH,  # Set working directory
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(f"Dataset cloned successfully to: {ORIGINAL_DATASET_PATH}")
                return True
            else:
                print(f"Failed to clone dataset. Error: {result.stderr}")
                return False

        except Exception as e:
            print(f" Error cloning dataset: {e}")
            return False

# DATASET ORGANIZATION

def copy_image(src_dst):
    """Helper function to copy image files."""
    src, dst = src_dst
    try:
        shutil.copy(src, dst)
    except Exception as e:
        print(f" {e}")


def create_multi_fruit_dataset():
    """Organize dataset into fresh/rotten categories for each fruit."""
    print("\nCreating multi-fruit dataset (OPTIMIZED)")

    # Reset output directory
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
        print(" Cleaned existing folder")

    # Create folder structure
    for split in ['Training', 'Test']:
        for fruit in FRUIT_CATEGORIES:
            for state in ['fresh', 'rotten']:
                os.makedirs(os.path.join(OUTPUT_PATH, split, f"{fruit.capitalize()}_{state}"))

    print(" Folder structure created from FRUIT_CATEGORIES")

    total_copied = 0

    for split in ['Training', 'Test']:
        print(f"\n Processing: {split}")
        split_path = Path(ORIGINAL_DATASET_PATH) / split

        if not split_path.exists():
            print(f" Split not found: {split_path}")
            continue

        all_dirs = [d for d in split_path.iterdir() if d.is_dir()]
        tasks = []

        for fruit in FRUIT_CATEGORIES:
            print(f"\n Processing fruit: {fruit.capitalize()}")

            valid_dirs = [d for d in all_dirs if d.name.startswith(fruit.capitalize())]
            print(f"   Found {len(valid_dirs)} classes")

            for class_dir in valid_dirs:
                images = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))

                # Determine if rotten or fresh
                if class_dir.name in FRUIT_CATEGORIES[fruit]['rotten']:
                    dest_class = f"{fruit.capitalize()}_rotten"
                else:
                    dest_class = f"{fruit.capitalize()}_fresh"

                dest_folder = Path(OUTPUT_PATH) / split / dest_class

                print(f"    {class_dir.name} → {dest_class} ({len(images)} images)")

                for img in images:
                    dst = dest_folder / f"{class_dir.name}_{img.name}"
                    tasks.append((str(img), str(dst)))

        print(f"\n Copying {len(tasks)} images for {split}...\n")

        with ThreadPoolExecutor(max_workers=12) as pool:
            list(tqdm(pool.map(copy_image, tasks), total=len(tasks), desc=split, unit="img"))

        total_copied += len(tasks)
        print(f"{split} finished: {len(tasks)} images copied")

    # Verification
    print("\n Verifying structure...\n")
    grand_total = 0

    for split in ['Training', 'Test']:
        split_path = Path(OUTPUT_PATH) / split
        split_total = 0

        for folder in split_path.iterdir():
            if folder.is_dir():
                count = len(list(folder.glob("*.jpg"))) + \
                        len(list(folder.glob("*.png"))) + \
                        len(list(folder.glob("*.jpeg")))

                print(f"  {folder.name}: {count}")
                split_total += count

        print(f"{split} TOTAL: {split_total}\n")
        grand_total += split_total

    print(f"\nALL DONE: {grand_total} images copied")
    print(f"Dataset located at: {OUTPUT_PATH}")

    return OUTPUT_PATH

# Data Generators
TRAIN_DIR = os.path.join(OUTPUT_PATH, 'Training')
TEST_DIR = os.path.join(OUTPUT_PATH, 'Test')
def create_generators():
    """
    Builds training, validation, and test data generators.
    """
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    print(f" Training directory: {TRAIN_DIR}")
    print(f" Test directory: {TEST_DIR}")
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=True
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    print(f"Classes: {train_gen.class_indices}")
    print(f" Training samples: {train_gen.samples}")
    print(f" Validation samples: {val_gen.samples}")
    print(f" Test samples: {test_gen.samples}")
    return train_gen, val_gen, test_gen


# Save Config

def save_config(train_gen, val_gen, test_gen):
    """Saves dataset configuration for the train script."""

    config = {
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        'validation_split': VALIDATION_SPLIT,
        'train_dir': TRAIN_DIR,
        'test_dir': TEST_DIR,
        "num_classes": len(train_gen.class_indices),
        "class_indices": train_gen.class_indices,
        "train_samples": train_gen.samples,
        "val_samples": val_gen.samples,
        "test_samples": test_gen.samples
    }

    with open(DATA_CONFIG_FILE, "wb") as f:
        pickle.dump(config, f)

    print(f"Configuration saved to  {DATA_CONFIG_FILE}")
    return config


# Main Execution


if __name__ == "__main__":

    print("Starting dataset preparation...\n")
    print("\nSTEP 1: DATASET DOWNLOAD & SETUP")
    dataset_ready = ensure_dataset_exists()
    if dataset_ready:
        print("\nSTEP 2: DATASET ORGANIZATION")
        dataset_path = create_multi_fruit_dataset()

        print("\nSTEP 3: DATA GENERATORS SETUP")
        train_gen, val_gen, test_gen = create_generators()

        print("\nSTEP 4: SAVE CONFIGURATION")
        config = save_config(train_gen, val_gen, test_gen)

        print("\nSummary:")
        print(f"Classes: {config['num_classes']}")
        print(f"Training samples: {config['train_samples']}")
        print(f"Validation samples: {config['val_samples']}")
        print(f"Test samples: {config['test_samples']}")
        print(f"\n Next step: Run 'model_training.py'")
