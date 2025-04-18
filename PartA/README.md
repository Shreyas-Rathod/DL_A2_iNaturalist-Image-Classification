## 🧠 Part A – Training a CNN from Scratch

---

### 📌 Objective

The goal of Part A is to:
- Build a **custom CNN model** from scratch  
- Tune its hyperparameters using **W&B Sweeps**  
- Analyze performance and generalization  
- Compare training dynamics across configurations

We used a subset of the **iNaturalist 12K dataset** with 10 classes.

---

### 🧱 Q1 – CNN Model Implementation (5 Marks)

We designed a modular CNN architecture with:

- **5 convolutional blocks**:  
  Each block = `Conv2D → Activation → MaxPool`
- **Configurable hyperparameters**:  
  - Filter size  
  - Number of filters  
  - Activation function (ReLU, GELU, SiLU...)  
  - Number of dense units

Final structure:
- 1 hidden `Linear` (dense) layer  
- 1 output layer with 10 neurons

✅ Also included:
- Code to calculate **total number of parameters**
- Estimation of **computational cost** (MACs)

---

### 🧪 Q2 – Training & Hyperparameter Tuning (15 Marks)

We split the dataset as:
- **80%** for training  
- **20%** from `train/` used for validation  
- Test data (`val/` folder) untouched for now

We used **Weights & Biases (W&B)** to run a sweep over:

| Hyperparameter | Values Tried |
|----------------|--------------|
| `num_filters` | 32, 64 |
| `activation` | ReLU, GELU, SiLU |
| `dropout` | 0.0, 0.2, 0.3 |
| `num_dense_neurons` | 128, 256 |
| `batch_norm` | True, False |
| `lr` | 1e-3, 5e-4 |

📊 W&B Automatically Logged:
- Accuracy vs Epochs  
- Correlation Summary Table  
- Parallel Coordinates Plot  

✅ Strategy:
- Used **Bayesian sweep** to reduce experiments while maximizing performance.

---

### 📈 Q3 – Observations from Sweep (15 Marks)

Key insights from 39 sweep runs:

- ✅ **BatchNorm ON** consistently improved accuracy (~9% higher)
- ✅ **Starting with 32 filters** performed better than 64/128/512
- ✅ **ReLU** outperformed GELU & SiLU in this architecture
- ✅ Best dropout for stability was around **0.2–0.3**
- ❌ Models with **too many filters in early layers** overfit easily
- ✅ LR = **1e-3** was effective; 5e-4 was too slow for convergence

---

### 🧪 Q4 – Test Evaluation & Visualizations (5 Marks)

We selected the **best model** from the sweep and evaluated it on the **unseen test set** (`val/` folder).

✅ Final Test Accuracy: **~40%**  
✅ Logged in W&B:
- 10×3 grid of test images with predictions  
- First-layer filters (8×8 grid)  
- Guided backprop (top 10 activations from CONV5)

---

### 🔗 Files & Notebooks

| File | Description |
|------|-------------|
| `model.py` | Custom CNN model definition |
| `train.py` | Training loop with wandb sweep support |
| `evaluate.py` | Final evaluation + plots |
| `sweep_config.yaml` | W&B sweep configuration |
| `README.md` | This file |

---

### 🛠️ Tools Used

- PyTorch + Torchvision  
- W&B for experiment tracking  
- Captum (optional) for interpretability  
- Kaggle GPU for training  

---

### 🏁 Summary

Part A demonstrated that:
- A small CNN can learn basic patterns with careful tuning  
- Hyperparameter sweeps reveal valuable insights  
- But pretrained models (see Part B) outperform scratch models on small datasets
