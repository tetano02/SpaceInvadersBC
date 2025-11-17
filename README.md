# CTRLC-CTRLPAC - Interactive Imitation Learning Project

## Overview

This project is part of the **Social Robotics – MU5EEH15 (2025/2026)** course.
It focuses on developing an **interactive imitation learning framework**, combining **Behavioral Cloning** and **DAgger**, aligned with the course objective of exploring human-in-the-loop reinforcement learning.

## Project Description - Phases (To be updated while the project progress)

- Our initial plan was to use the **Pac-Man (ALE/Pacman-v5)** environment from Gymnasium, as it provides a visually intuitive and partially stochastic scenario (due to ghost movement).  
  However, we realized that the **state space of Pac-Man is highly complex and high-dimensional**, making it challenging to collect and manage demonstrations effectively in our current setup.

- To simplify training and facilitate faster experimentation, we decided to switch to the **Space Invaders (ALE/SpaceInvaders-v5)** environment.  
  This environment maintains visual control and temporal dynamics similar to Pac-Man, while offering a **more manageable observation space**, allowing us to focus on the imitation learning aspects rather than extensive state abstraction.

#### 12/11

- First try: One demonstration -> The agent learns only to move left and right, not even properly and it doesn't shoot. Possible reason: LEFT/RIGHT/NOPE -> Press continously. FIRE -> One press. It is more probable that it learns the movement behavior.

- Prova con test esperto Minari: Dataset esperto - Epochs : 50 - Batch size: 32

#### Continue

## Objectives

- Implement **Behavioral Cloning** to train an initial policy from human demonstrations (via keyboard teleoperation).
- Extend the model with **DAgger (Dataset Aggregation)** to iteratively improve policy performance using interactive corrections.
- Evaluate the effectiveness of human-in-the-loop training in a visual control task.

## Methodology

1. **Data Collection:** Human demonstrations recorded via Gymnasium environment wrappers.
2. **Behavioral Cloning:** Train a neural policy network to imitate human trajectories.
3. **DAgger Iterations:** Run interactive sessions where a human provides corrective actions as the agent plays.
4. **Evaluation:** Compare performance before and after DAgger corrections.

## Tools and Frameworks

- **Gymnasium (Farama Foundation)** – Environment simulation
- **Stable-Baselines3 / PyTorch** – Reinforcement learning backbone
- **Minari** – Dataset handling for offline RL and imitation learning
- **Custom Keyboard Teleoperation Script** – For demonstration collection

## Environment Setup

The repo uses **Python 3.13** and [**uv**](https://docs.astral.sh/uv/) for dependency management. Follow these steps to get a working workstation, including GPU support.

### 1. System prerequisites (GPU)

- NVIDIA GPU with recent drivers (≥ the game-ready/Studio release that supports CUDA 12.4).
- (Optional) CUDA Toolkit 12.4 — not strictly necessary, but installing it guarantees the matching runtime libraries. The PyTorch wheels we rely on already ship the needed CUDA libs.
- Verify that Windows sees your GPU: `nvidia-smi` should list the device without errors.

### 2. Python + uv

1. Install Python 3.13 from [python.org](https://www.python.org/downloads/) and allow the installer to add it to `PATH`.
2. Install uv (once):
   ```powershell
   pip install uv
   uv --version
   ```

### 3. Clone & sync the project

```powershell
git clone https://github.com/tetano02/CTRLC-CTRLPAC.git
cd CTRLC-CTRLPAC
uv sync
```

`uv sync` creates a `.venv` (or reuses your configured interpreter) and installs the dependencies listed in `pyproject.toml`. The PyTorch entries point to the CUDA build, so we must install them using the PyTorch wheel index in the next step.

### 4. Install CUDA-enabled PyTorch wheels

1. Remove any CPU-only builds that may have been pulled in automatically:
   ```powershell
   uv pip uninstall torch torchvision torchaudio
   ```
2. Reinstall the CUDA 12.4 wheels from the official PyTorch index:

   ```powershell
   uv add torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 `
     --index https://pypi.org/simple `
     --extra-index-url https://download.pytorch.org/whl/cu124 `
     --index-strategy unsafe-best-match
   ```

   - `--index` keeps PyPI as the primary source for the rest of the dependencies.
   - `--extra-index-url` exposes the CUDA wheels published by PyTorch.
   - If you already trust both indexes equally, consider adding the same values to `.uv/config.toml` to avoid retyping them.

3. Confirm that PyTorch detects your GPU:
   ```powershell
   uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```
   Expected output:
   - Version ends with `+cu124` (e.g., `2.6.0+cu124`).
   - `torch.cuda.is_available()` prints `True`.
   - Your GPU name (e.g., `Quadro T2000 with Max-Q Design`).

From here you can launch the CLI workflows, e.g. `uv run main.py`, collect demos (`collect_demonstrations.py`), load Minari experts, and start Behavioral Cloning training on GPU.

### Minari dataset storage

- Durante l'esecuzione di `load_minari_dataset.py` il progetto imposta automaticamente la variabile `MINARI_DATASETS_PATH` su `data/minari_datasets/`.
- Se vuoi usare un percorso diverso, definisci `MINARI_DATASETS_PATH` prima di avviare lo script; il valore personalizzato verrà rispettato.
- La cartella viene creata automaticamente e rimane versionata solo localmente (è ignorata da git), così i dataset scaricati non escono dal repository.

## Expected Results

We expect the agent trained with DAgger to outperform the Behavioral Cloning baseline, particularly in handling previously unseen game states and improving robustness.

## Metrics Export

Every Behavioral Cloning run now produces a synchronized CSV report in `data/metrics/`. Each file groups three sections:

- **Metadata:** environment name, training/evaluation IDs, datetime, number/names of demonstrations, dataset split, epoch/batch parameters, and total training duration.
- **Epoch Metrics:** per-epoch train/validation loss and accuracy so the plots can be replicated or post-processed without screenshots.
- **Evaluation:** when `evaluate_bc.py` is executed, the corresponding CSV is automatically appended with summary statistics and per-episode rewards/lengths.

The CSV shares the same timestamp/ID used for the model checkpoint and plots, making it easy to trace back the assets belonging to the same experiment.

## Group Information

**Group Name:** CTRL+C - CTRL+PAC
**Members:** Agnelli Stefano, Cremonesi Andrea, Mombelli Tommaso, Sun Wen Wen

uv pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124

uv run python -c "import torch;print(torch.**version**);print(torch.version.cuda);print(torch.cuda.is_available());print(torch.cuda.device_count());\nprint([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
