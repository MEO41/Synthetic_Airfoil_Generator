# Neural Networks Reinforced Transformers Pipeline for Synthetic Airfoil Generation ✈️

This repository implements a full pipeline for generating synthetic airfoils using a Variational Autoencoder (VAE)-based architecture, with advanced sampling, diversity analysis, and visual inspection capabilities.

---

## 🧠 Pipeline Overview

The project enables:
- Learning compact latent representations of airfoil shapes
- Generating novel, diverse, and valid airfoil geometries
- Evaluating their uniqueness and distribution

A schematic of the pipeline:

![Pipeline Diagram](https://github.com/MEO41/Synthetic_Airfoil_Generator/blob/main/assets/A_flowchart_style_digital_illustration_diagram_ill.png?raw=true)


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
  ```
  - Supports β-VAE configurations for disentangled latent spaces

---

## 🚀 Training

- Early stopping with patience and learning rate scheduling
- `train()` method manages full training lifecycle, including saving best-performing model (`best_airfoil_model.pt`)

---

## 🧬 Generation & Diversity

### 🌱 Sampling & Generation
- `generate_airfoil()`: Sample from latent space
- `generate_multiple_airfoils(n)`: Batch generation
- `interpolate_airfoils(z1, z2)`: Latent space interpolation between two airfoils

### ✅ Quality Validation
- `validate_airfoil_quality()`: Checks x/y range and geometric smoothness

### 📊 Diversity Analysis
- `analyze_diversity()`: Computes pairwise Euclidean distances
- Compares generated airfoils to original dataset using internal diversity metrics and novelty score

---
![Example Output Which is not optimized yet](https://github.com/MEO41/Synthetic_Airfoil_Generator/blob/main/assets/example_output.jpg?raw=true)
## 📦 Folder Structure

```
.
├── airfoils/                 # Raw .dat airfoil data !!! our data set is not been published to github yet
├── generated_airfoils/      # Saved output samples
├── A_flowchart_style_digital_illustration_diagram_ill.png
├── README.md
└── main.py
```

---

## 🛠 Requirements

```bash
pip install torch numpy scikit-learn matplotlib scipy
```

---

## 📌 Citation

If you use this project, please cite as:

```
ONAY, M. (2025). Synthetic Airfoil Generation using Neural Reinforced Transformers [Code]. GitHub.
```

---

## 🧠 Future Work

- Integration of Transformer layers in latent space modeling
- Conditional airfoil generation (e.g., based on performance metrics) + neuralairfoil
- CFD or surrogate model coupling for fitness-based filtering

---

## 📬 Contact

Maintained by **Muhammed Emin ONAY**  
[LinkedIn](www.linkedin.com/in/muhammed-emin-onay-02328129a) • [GitHub](https://github.com/MEO41)

---
