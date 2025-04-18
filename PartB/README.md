## 🧠 Part B – Fine-tuning a Pretrained Model

---

### 📌 Objective

In real-world applications, training from scratch is often impractical.  
In Part B, we fine-tune a **large model pre-trained on ImageNet** using the iNaturalist dataset (10 classes).

We explored:
- Adapting image size and normalization
- Replacing final classification layer
- Layer-freezing strategies to balance efficiency and performance

---

### 🔧 Q1 – Implementation Challenges (5 Marks)

| Problem | Solution |
|--------|----------|
| **Image size mismatch** | Resized all input images to **224×224** (ImageNet standard) |
| **Normalization mismatch** | Applied **ImageNet mean/std** normalization |
| **Output layer mismatch** | Replaced final classification layer with `nn.Linear(..., 10)` for 10-class output |

✅ We used **ResNet-50** as the pretrained model.  
✅ All layers except the final layer were initialized with **ImageNet weights** via `torchvision.models`.

---

### 🔁 Q2 – Freezing Strategies (5 Marks)

We tried **3 common fine-tuning strategies** to keep training efficient and reduce overfitting:

#### 🔹 Strategy 1 – Freeze all layers except final FC
- Only trains the new `nn.Linear` output layer.
- Fastest and best for small datasets.

#### 🔹 Strategy 2 – Freeze up to `layer3`, fine-tune `layer4` and `fc`
- Keeps low-level features fixed and adapts high-level ones.
- Good balance between compute and performance.

#### 🔹 Strategy 3 – Train all layers
- Fully updates the model.
- High risk of overfitting on small datasets.

✅ All experiments were logged using **Weights & Biases (W&B)**.

---

### 📊 Q3 – Sweep Results & Insights (5 Marks)

We ran a W&B sweep with the above 3 strategies. Here's what we observed:

| Strategy | Val Acc | Val Loss | Train Acc | Sweep Name |
|----------|---------|----------|-----------|------------|
| Freeze all but FC      | **77.5%** | **0.695**  | 75.7%     | `revived-sweep-1` |
| Freeze until `layer4`  | 76.4%     | 0.849     | **94.4%** | `cerulean-sweep-2` |
| Train all layers       | 61.2%     | 1.38      | 79.3%     | `bright-sweep-3` |

#### 🔍 Key Takeaways

- ✅ **Freezing most layers** led to **best generalization** and **lowest validation loss**.
- ❌ **Training all layers** overfit quickly despite high training accuracy.
- ✅ Even the weakest fine-tuned model **outperformed training-from-scratch** (Part A) by **30–35%** on val accuracy.

---

### ✅ Q4 – Final Fine-tuning Strategy + Comparison (5 Marks)

We chose **Strategy 2 (freeze until layer4)** for final fine-tuning.  
Compared to Part A (scratch-trained CNN with ~40% val accuracy), our fine-tuned ResNet50 achieved:

- **94.4% training accuracy**
- **76.4% validation accuracy**
- **Faster convergence (5 epochs)**

---

### 🧪 Tools & Libraries

| Tool | Purpose |
|------|---------|
| `torchvision.models.resnet50(pretrained=True)` | Load ImageNet weights |
| W&B Sweeps | Compare strategies |
| Captum (optional) | Interpretability |
| Kaggle GPU | Training environment |

---

### 🔗 Key Files

| File | Description |
|------|-------------|
| `finetune_resnet.py` | Fine-tunes ResNet50 with W&B logging |
| `finetune_sweep.py` | Runs sweep across freezing strategies |
| `partB_eval.ipynb` | Final model eval and sample predictions |
| `README.md` | This file |

---

### 🏁 Summary

Part B shows how **transfer learning** with layer-freezing provides:
- Better accuracy
- Faster training
- Lower risk of overfitting

Compared to training from scratch, it’s significantly more efficient and practical for small to medium datasets.
