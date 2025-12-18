# CS417 Project – Repository Walkthrough & Usage Guide

This document explains **how the repository works**, how its scripts depend on each other, and the **correct execution order** to either **train a new model** or **evaluate an existing one**.

The project focuses on **fruit image classification** using a **controlled subset of the Fruit-360 dataset**, with reproducible preprocessing, configurable class selection, and modular training/evaluation pipelines.

---

## 1. Prerequisites

Before doing anything else, make sure you have the following installed:

- **Python 3.9 or higher**
- `git`
- A GPU-enabled environment **(recommended for training)** such as:
  - Kaggle
  - Google Colab
  - Lightning AI
  - Any local CUDA-capable setup (e.g., T4 GPU with ~12 GB VRAM)

> ⚠️ CPU-only environments will work for **evaluation**, but training is **strongly discouraged** without a GPU.

---

## 2. Clone the Repository

Clone the project using Git:

```bash
git clone https://github.com/Muhammad-H-Bakr/CS417_Project.git
cd CS417_Project
```

All scripts assume execution **from inside the repository root**.

---

## 3. Environment Setup

You have **two supported ways** to install dependencies.

### Option A – Automated Setup (Recommended)

Run the setup script:

```bash
python setup.py
```

This script will:

- Create a local `.venv`
- Guide you to Activate it
- Install all required dependencies

Follow the on-screen instructions carefully.

---

### Option B – Manual Installation

If you prefer to manage environments yourself (or the environment you work with does not allow the use of .venv):

```bash
pip install -r requirements.txt
```

Make sure the environment you install into is the same one you use to run all scripts.

---

## 4. Data Preprocessing (MANDATORY STEP)

### Script: `data_preprocessing.py`

This is the **most critical script in the entire repository**.

You **must run it first**, regardless of whether you want to:

- Train a model
- Evaluate a trained model

```bash
python data_preprocessing.py
```

(or run a notebook including its code — both behave identically)

---

### What This Script Does

Running `data_preprocessing.py` will:

1. **Clone the original fruits-360-100x100 dataset**
2. **Create a project-scoped subset** of the dataset (multi_fruit_freshness_dataset)
3. Apply class filtering based on a custom label dictionary
4. Generate a configuration file:
   - `data_config.pkl`
5. Ensure all dataset paths are **absolute and environment-aware**

---

### Why It Must Be Re-Run Per Environment

This script:

- Detects the **absolute project path**
- Stores environment-specific paths in `data_config.pkl`

Therefore, it **must be executed again** when switching between:

- Local machine
- Google Colab
- Kaggle
- Lightning AI
- Any other execution environment

> ⚠️ Skipping this step will cause **path resolution errors** later.

---

### Custom Class Control – `FRUIT_CATEGORIES`

The dataset scope is defined by the dictionary:

```python
FRUIT_CATEGORIES = { ... }
```

This dictionary:

- Acts as a **hand-crafted label map**
- Controls which fruits are included
- Defines the final classification task

#### Scalability Note

To change the task:

1. Modify `FRUIT_CATEGORIES`
2. Re-run `data_preprocessing.py`
3. Re-run model training

No other code changes are required.

---

## 5. Model Training (Optional)

### Script: `model_training.py`

If you want to **train the model yourself**, run:

```bash
python model_training.py
```

(or use the notebook version)

---

### Training Requirements

- GPU recommended (T4 12 GB or similar)
- `data_preprocessing.py` **must have been run first**

The training script consumes:

- The dataset subset
- `data_config.pkl`
- The class structure defined in `FRUIT_CATEGORIES`

---

### Training Outputs

After training completes, the following files are generated:

- **Model file**: `best_fruit_freshness_model.keras`
- **Training history**: `training_history.pkl`

These files are later required for evaluation.

---

## 6. Evaluation Only (Using Pretrained Model)

If you are **not interested in training** and only want results:

1. Run preprocessing:

   ```bash
   python data_preprocessing.py
   ```

2. Ensure that you have (or the repository already contains):
   - A trained `.keras` model (respective to its data preprocessing)
   - Its corresponding `.pkl` history file
3. Run evaluation:

   ```bash
   python evaluation_testing.py
   ```

---

## 7. Evaluation Outputs

After evaluation finishes, **four output files** are produced:

1. **Confusion Matrix**
2. **Training History Visualization**
3. **Model Predictions**
4. **Evaluation Summary** (metrics & performance report)

These files represent the **final deliverables** of the project.

---

## 8. Execution Summary (Quick Reference)

### One-Time Per Environment (or Classification scope change)

```text
data_preprocessing.py
```

### Then Either

### **Train + Evaluate**

```text
model_training.py
evaluation_testing.py
```

### **OR**

### **Evaluate Only (trained Model)**

```text
evaluation_testing.py
```

> 🔁 You only need to re-run preprocessing if:
>
> - You change environments
> - You modify `FRUIT_CATEGORIES`

---

## Final Notes

- Script order matters
- Preprocessing is the backbone of reproducibility
- Class scalability is intentionally centralized
- Training and evaluation are cleanly decoupled
