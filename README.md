# SpaceInvadersBC – Interactive Imitation Learning on Space Invaders

This repository contains the final project for the course **Social Robotics – MU5EEH15 (2025/2026)**.  
The goal is to explore **imitation learning** in the Atari game *Space Invaders*, starting from **Behavioral Cloning (BC)** and extending it with **GAIL** for adversarial reward refinement.  
All models operate directly on **raw RGB frames** from `ALE/SpaceInvaders-v5`.

---

## 1. Project Overview

This project implements the full imitation-learning pipeline:

1. **Loading Minari expert demonstrations** as the only dataset.
2. **Training Behavioral Cloning (BC)** models with three architectures:
   - **MLP**
   - **CNN / DQN-style network**
   - **Vision Transformer (ViT)**
3. **Logging metrics** (loss, accuracy, metadata) into CSV files.
4. **Evaluating trained agents** on multiple Space Invaders episodes.
5. **Experimenting with GAIL** to improve imitation quality beyond pure BC.

Everything is implemented in **Python + PyTorch**.

---

## 2. Repository Structure

### `main.py`
Entry point of the project. Provides a menu to:
- train BC models,
- load Minari data,
- view existing models/metrics,
- evaluate saved policies.

### `behavioral_cloning.py`
Core BC implementation:
- dataset/dataloader handling,
- three model architectures (MLP, CNN, ViT),
- training loop with validation,
- checkpoint saving,
- CSV logging.

### `data_manager.py`
Handles all data operations:
- loading Minari expert trajectories,
- converting them to internal `.pkl` format,
- splitting train/validation sets.

### `env_make.py`
Creates the `ALE/SpaceInvaders-v5` environment:
- applies wrappers,
- handles seeding,
- performs preprocessing.

### `test_model.py`
Evaluates a trained model on several episodes and updates the metrics CSV.

### `gail/`
Contains experimental GAIL components:
- discriminator network,
- reward extraction utilities,
- evaluation scripts.

### `plot_results/`
Scripts & notebooks for plotting:
- learning curves,
- reward distributions,
- architecture/dataset comparisons.

### `pyproject.toml` / `uv.lock`
Dependency and environment management via `uv`.

---
## 3. Datasets

The project supports **two sources of demonstrations**:

### 3.1 Human Demonstrations
The code includes a built-in teleoperation tool that allows the user to **record their own gameplay** using the keyboard.  
These demonstrations are saved as `.pkl` files and follow the same structure used for training:
- raw RGB frames,
- the selected action,
- reward information.

Human demos can be collected at any time through the main script.

### 3.2 Minari Expert Dataset
The project can automatically **download and convert the Minari Space Invaders expert dataset**.  
A dedicated script loads the dataset and transforms it into the same `.pkl` format used for BC training.

Using the two data sources, the training pipeline allows:
- human-only training,
- Minari-only training,
- or **mixed datasets** combining both.

---

## 4. Behavioral Cloning Pipeline

1. **Load Minari dataset** through `DataManager`.
2. **Split** into:
   - 80% training,
   - 20% validation.
3. **Choose a model architecture** (MLP, CNN, ViT).
4. **Train for 50 epochs** using batch sizes:
   - 32, 64, 128, or 256.
5. **Track validation loss** to save the best-performing checkpoint.
6. **Log everything** to a CSV:
   - losses,
   - accuracies,
   - architecture info,
   - dataset info,
   - timestamps.

This enables systematic comparison across architectures.

---

## 5. Evaluation

`test_model.py` runs a trained model over multiple episodes and computes:

- mean reward,
- standard deviation,
- episode lengths.

These results are appended to the original metrics CSV to keep all experiment data synchronized.

This allows comparison between:

- MLP vs CNN vs ViT,
- BC vs GAIL-refined models.

---

## 6. GAIL (Experimental Extension)

The `gail/` module implements a simplified GAIL pipeline:

- A **discriminator** distinguishes expert vs agent `(state, action)` pairs.
- The discriminator output is transformed into a **learned reward**.
- The policy is then trained (or fine-tuned) using this adversarial reward.

This is used to explore whether GAIL helps the model:
- recover from underrepresented states,
- improve generalization,
- reduce imitation bias.

---

## 7. Environment Setup

This project is configured for **Python 3.13** and **`uv`**.

### 7.1 Installing with `uv`

```bash
pip install uv
git clone https://github.com/tetano02/SpaceInvadersBC.git
cd SpaceInvadersBC
uv sync
````

### (Optional) Test PyTorch installation

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 8. Usage Guide

### 8.1 Load Minari Expert Dataset

```bash
uv run load_minari_dataset.py
```

### 8.2 Train a BC Model

```bash
uv run main.py
```

Choose:

* dataset: Minari
* architecture: MLP / CNN / ViT
* batch size and number of epochs

### 8.3 Evaluate a Model

```bash
uv run test_model.py --model-path path/to/model.pt
```

### 8.4 Plot Results

Use scripts in:

```
plot_results/
```

---

## 9. Expected Behavior

* Minari provides strong teacher behavior.
* CNN and ViT outperform MLP due to visual structure.
* GAIL may improve robustness in rarely-seen states.
* Mixed architecture results can highlight generalization limits.

---

## 10. Group

**SpaceInvadersBC** *(formerly CTRL+C – CTRL+PAC)*

* Agnelli Stefano
* Cremonesi Andrea
* Mombelli Tommaso
* Sun Wen Wen
