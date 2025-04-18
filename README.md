# 🌿 iNaturalist Image Classification  
### Deep Learning Assignment 2 – Train from Scratch + Fine-tune Pretrained Models  

---
**Student Info : Shreyas Rathod (CS24M046)**

**Course Instructor** : Mitesh Khapra

---
## Objective  
This assignment explores two approaches for image classification using a subset of the [iNaturalist 12K dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip):  
- **Part A:** Train a custom CNN from scratch  
- **Part B:** Fine-tune a pretrained model (e.g., ResNet50)  
All experiments are tracked using **Weights & Biases (wandb)**.

---

## 📁 Dataset Structure  
```
inaturalist_12K/
├── train/        # 80% train + 20% val split used internally
└── val/          # Held-out test set (not used in training)
```

---

## 📦 Part A – Custom CNN from Scratch  

### ✅ Q1: Flexible CNN  
- 5 Conv → Activation → MaxPool blocks  
- Configurable: filters, activations, dropout, dense layers  
- Ends with `Linear → Linear(10)` for 10 classes  

### ✅ Q2: Training with W&B Sweeps  
We tuned:  
`num_filters`, `activation`, `dropout`, `batch_norm`, `num_dense_neurons`, `lr`  
→ Used Bayesian sweeps to reduce trials.

### ✅ Q3: Observations  
- **BatchNorm ON** improved accuracy (~9%)  
- **32 filters** outperformed 64/128/512  
- **ReLU** > GELU/SiLU  
- **Best val acc ~40%**

### ✅ Q4: Final Evaluation  
- Logged test accuracy, filter visualizations, and guided backprop using Captum  
- **Test accuracy ~40%**

---

## 📦 Part B – Fine-tuning Pretrained ResNet50  

### ✅ Q1: Adjustments  
- Resized inputs to **224×224**  
- Replaced final `Linear(1000)` → `Linear(10)`  
- Used ImageNet normalization stats  

### ✅ Q2: Layer Freezing Strategies  
Tried 3 approaches:  
1. Freeze all except final FC  
2. Freeze until `layer3`, train `layer4 + fc` ✅  
3. Train all layers  

→ Sweeps run using wandb to compare  

### ✅ Q3: Insights from Experiments  
| Strategy               | Val Acc | Notes                         |
|------------------------|---------|-------------------------------|
| Freeze all but FC      | **77.5%** | Best generalization           |
| Freeze until layer4    | 76.4%   | Balanced compute + accuracy   |
| Train all layers       | 61.2%   | Overfit on small dataset      |
| From Scratch (Part A)  | 40.0%   | Baseline                      |

✅ Fine-tuned models converged faster and performed **30%+ better** than from-scratch training.

---

## 📂 Tools & Libraries  
- `PyTorch`, `torchvision`, `wandb`, `captum`, `Kaggle GPU`

---

## 📂 Key Files  
| File | Description |
|------|-------------|
| `CNN_Model.py` / `TrainLoop_wandb.py` | Custom CNN definition & trainer |
| `finetune_resnet.py` | Fine-tunes ResNet50 |
| `finetune_sweep.py` | W&B sweep over freezing strategies |
| `final_eval_plots.py` | Final test evaluation & visualization |
| `README.md` | This summary |

---
## 📁 Wandb Report

---
## 🏁 Final Thoughts  
> **Transfer learning with strategic layer freezing provides a huge boost in accuracy, stability, and training speed—especially when data is limited.**
