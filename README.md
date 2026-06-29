# Vision-X & GAN Remote-Sensing Platform: Developer Guide

This repository contains a full-featured PyTorch machine learning codebase for paired and unpaired image-to-image translation (Pix2Pix and CycleGAN) tailored for satellite remote-sensing tasks (such as the Guwahati Azara dataset). It includes a database-driven graphical user interface called the **Vision-X Dashboard** for managing training pipelines, data preprocessing, log streaming, and interactive testing.

---

## Table of Contents
1. [Core Architecture & File Structure](#1-core-architecture--file-structure)
2. [Environment Setup & Installation](#2-environment-setup--file-structure)
3. [Vision-X Dashboard GUI Manual](#3-vision-x-dashboard-gui-manual)
4. [Training Run Database Schema](#4-training-run-database-schema)
5. [Command-Line Interface CLI](#5-command-line-interface-cli)
6. [Standalone Binary Compilation](#6-standalone-binary-compilation)
7. [Automated Scripts Directory](#7-automated-scripts-directory)

---

## 1. Core Architecture & File Structure

The project is structured modularly. The main application files and folders are defined below:

*   **Main Application Controllers:**
    *   [gui.py](file:///home/sankhya/Coding/Python/GAN-ml-/gui.py): Main entrypoint and layout for the CustomTkinter GUI dashboard.
    *   [build_gui.py](file:///home/sankhya/Coding/Python/GAN-ml-/build_gui.py): Packages the GUI, including CustomTkinter themes and DLLs/assets, into a single executable binary.
    *   [start_gui.sh](file:///home/sankhya/Coding/Python/GAN-ml-/start_gui.sh): Helper shell script to launch the GUI using the pre-configured Python interpreter.
    *   [train.py](file:///home/sankhya/Coding/Python/GAN-ml-/train.py): CLI training entrypoint for training GAN models.
    *   [test.py](file:///home/sankhya/Coding/Python/GAN-ml-/test.py): CLI testing/inference entrypoint for GAN models.
    *   [database.db](file:///home/sankhya/Coding/Python/GAN-ml-/database.db): Local SQLite database managing training history and progress logs.

*   **Submodules:**
    *   [models/](file:///home/sankhya/Coding/Python/GAN-ml-/models): Architecture specifications, optimization routines, and neural network generator/discriminator models.
        *   `base_model.py`: Abstract class defining common workflows (saving, loading, forward passes).
        *   `pix2pix_model.py`: Paired conditional GAN model implementation.
        *   `cycle_gan_model.py`: Unpaired cycle-consistent GAN model implementation.
        *   `networks.py`: Network building blocks (ResNet, U-Net, PatchGAN discriminator, loss configurations).
    *   [data/](file:///home/sankhya/Coding/Python/GAN-ml-/data): Dataset managers and image loading pipelines.
        *   `base_dataset.py`: Abstract dataset base class.
        *   `aligned_dataset.py`: Loads paired (A and B aligned side-by-side) images for Pix2Pix.
        *   `unaligned_dataset.py`: Loads unpaired images for CycleGAN.
        *   `image_folder.py`: Internal helpers to query images from directories.
    *   [options/](file:///home/sankhya/Coding/Python/GAN-ml-/options): Settings and arguments parser.
        *   `base_options.py`: Shared parameters (dataroot, model, batch size, network sizes).
        *   `train_options.py`: Specific parameters for training (epochs, learning rates, decay scheduling).
        *   `test_options.py`: Specific parameters for inference and evaluation.
    *   [util/](file:///home/sankhya/Coding/Python/GAN-ml-/util): Helper utilities for visualizations, logging, database queries, and HTML reporting.

---

## 2. Environment Setup & Installation

You can set up your environment by compiling dependencies using Conda or standard Pip environments.

### Conda Setup
An environment file [environment.yml](file:///home/sankhya/Coding/Python/GAN-ml-/environment.yml) is included:
```bash
conda env create -f environment.yml
conda activate pytorch-img2img
```

### Manual Dependency Installation
To install the necessary python packages manually:
```bash
pip install torch torchvision torchaudio
pip install customtkinter pillow sqlite3
pip install visdom dominate wandb
```

---

## 3. Vision-X Dashboard GUI Manual

The dashboard provides a mouse-controlled frontend for all deep learning workflows:

1.  **Dashboard Home Page:** Displays overall system status (e.g., active GPUs, CPU tracking) and presents cards to launch individual workflow modules.
2.  **Train New Model:** Allows visual configuration of dataset root, learning rate, decay epochs, batch size, model framework (Pix2Pix or CycleGAN), and WandB integration. Clicking run executes `train.py` asynchronously.
3.  **Fine-tune Checkpoint:** Allows you to load previous weights, specify a parent run, and continue training from where you left off.
4.  **Monitoring & Real-time Logs:** Connects to the SQLite tracking database and shows active subprocess outputs. Logs stream directly inside the GUI.
5.  **Model Storage Explorer:** Opens a workspace window displaying checkpoints, listing individual `.pth` generator/discriminator files, and showing their disk size.
6.  **Dataset Preprocessing:** Focuses on satellite scene remote-sensing data (like the Guwahati Azara dataset). Allows cropping, tiling, and aligning satellite channels into model-ready paired images.
7.  **Interactive Model Tester:** Enables visual testing. Load a trained generator weight (`.pth`), select a local input image, and immediately run a prediction to view input vs. prediction maps side-by-side.

---

## 4. Training Run Database Schema

The dashboard tracks background training tasks using a SQLite database [database.db](file:///home/sankhya/Coding/Python/GAN-ml-/database.db). This ensures run data is not lost if the GUI is restarted.

### Table: `training_runs`
*   `id`: Primary integer key (auto-incrementing).
*   `name`: Label assigned to the training run.
*   `model`: GAN type (`pix2pix`, `cycle_gan`).
*   `direction`: Translation direction (`AtoB`, `BtoA`).
*   `netG`: Generator architecture layout (e.g., `unet_256`, `resnet_9blocks`).
*   `dataset_mode`: Dataset pairing type (`aligned`, `unaligned`).
*   `norm`: Normalization strategy (`batch`, `instance`, etc.).
*   `batch_size`: Batch size integer.
*   `n_epochs`: Standard learning rate epoch count.
*   `n_epochs_decay`: Linearly decaying learning rate epoch count.
*   `dataroot`: Directory path where datasets are located.
*   `gpu_ids`: Device ids allocated for training (e.g., `0`, `0,1`).
*   `use_wandb`: Flag (0 or 1) indicating weights & biases logging status.
*   `epoch_count`: Active epoch marker.
*   `status`: Operational state (`pending`, `running`, `completed`, `failed`).
*   `pid`: Process ID of the asynchronous python backend command.
*   `created_at`: Datetime stamp of initialization.
*   `completed_at`: Datetime stamp of process termination.
*   `log_file`: Relative file path to the standard output file.
*   `is_finetuning`: Checkbox flag for resumed training tasks.
*   `parent_run_id`: DB identifier of the model source run (if fine-tuning).
*   `parent_epoch`: Epoch loaded from the source model.

---

## 5. Command-Line Interface CLI

For users who prefer operating via the terminal, the Python files can be invoked directly:

### Training Options
Run `train.py` with standard flags:
```bash
python train.py --dataroot ./datasets/guwahati_azara_processed --name satellite_run --model pix2pix --direction AtoB --gpu_ids 0
```
Key Flags:
*   `--dataroot`: Directory path to the target image dataset.
*   `--model`: Model type (`pix2pix`, `cycle_gan`, `test`).
*   `--netG`: Generator architecture (`unet_128`, `unet_256`, `resnet_9blocks`).
*   `--gpu_ids`: CUDA device configuration (use `-1` for CPU training).
*   `--use_wandb`: Enable online monitoring logging.

### Testing Options
Run `test.py` to evaluate validation datasets:
```bash
python test.py --dataroot ./datasets/guwahati_azara_processed/test --name satellite_run --model pix2pix --direction AtoB
```
Results will save automatically to `./results/{name}/latest_test/`.

---

## 6. Standalone Binary Compilation

If you want to package the GUI into a single standalone program (executable binary) that runs without requiring python or pytorch installed on the target machine:

1.  Run the packaging script:
    ```bash
    python build_gui.py
    ```
2.  The script automatically detects the virtual environment's `customtkinter` assets (JSON templates, fonts) and invokes PyInstaller.
3.  The outputs are written to the `dist/` folder:
    *   **Linux:** `dist/Vision-X_Dashboard` (ELF binary)
    *   **Windows:** `dist/Vision-X_Dashboard.exe` (PE executable, must be built on Windows)

---

## 7. Automated Scripts Directory

The [scripts/](file:///home/sankhya/Coding/Python/GAN-ml-/scripts) folder contains automated utility pipelines:
*   `install_deps.sh`: Fast dependency installation wrapper.
*   `conda_deps.sh`: Conda environment creation script.
*   `train_pix2pix.sh`: Quick template script to start local pix2pix model training.
*   `test_pix2pix.sh`: Template to validate model outputs.
*   `train_satellite.sh`: Custom training script designed for multi-spectral remote-sensing datasets.
*   `test_before_push.py`: Code quality assurance script running flake8 linting and validation tests.
