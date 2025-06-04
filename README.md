# Neural Networks Reinforced Transformers Pipeline for Synthetic Airfoil Generation ✈️

This repository implements a full pipeline for generating synthetic airfoils using a Variational Autoencoder (VAE)-based architecture, with advanced sampling, diversity analysis, and visual inspection capabilities.

---

## 🧠 Pipeline Overview

The project enables:
- Learning compact latent representations of airfoil shapes
- Generating novel, diverse, and valid airfoil geometries
- Evaluating their uniqueness and distribution

A schematic of the pipeline:

![Pipeline Diagram](asssets/A_flowchart_style_digital_illustration_diagram_ill.png)

---

## 🧩 Components

### 📁 Data Loading & Resampling
- `load_dat_file(filepath)`: Parses `.dat` files containing airfoil coordinates.
- `resample_airfoil(coords, target_points)`: Resamples using arc-length parametrization to enforce a consistent shape resolution.

### 📏 Normalization
- A global `MinMaxScaler` is fit across the entire dataset to normalize the airfoil shapes into a compact domain.

### 🔀 Train/Validation Split
- `train_test_split` is used to separate the dataset into training and validation sets for robust model evaluation.

---

## 🧮 Model Architecture

### ⚙️ VAE Architecture (`AirfoilVAE`)
- Encoder → Latent space (μ, logσ²) → Decoder
- 3-layer fully connected architecture with dropout regularization

### 📉 Loss Function with Adjustable β
- Combined MSE reconstruction loss and KL divergence:
  ```python
  Loss = MSE(x, x_hat) + β * KL(mu, logvar)
