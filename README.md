## 🧠 CNN Assignment (Part A) – iNaturalist Dataset  
**Course:** Deep Learning  
**Objective:**  
1. Build and train a CNN model **from scratch**  
2. Tune hyperparameters using **W&B Sweeps**  
3. Analyze the results  
4. Evaluate the **best model** on unseen test data  

---

## 📁 Dataset  
We used a **subset of the [iNaturalist 12K dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip)**.  
The dataset was structured as:

```
inaturalist_12K/
├── train/        # Used for training + validation (80/20 split)
└── val/          # Used only for final testing
```

---

## 🚀 Q1 – Model Architecture from Scratch (5 Marks)

We implemented a configurable CNN with:
- **5 convolution blocks** (Conv → Activation → MaxPool)
- Support for **customizable**:  
  - filter size  
  - number of filters  
  - activation function (ReLU, GELU, SiLU...)  
  - dense layer size

The model ends with:
- 1 fully-connected dense layer  
- 1 output layer with 10 neurons (1 per class)

📌 Also included:
- Code to compute **total number of computations and parameters** given input image size and layer specs.

---

## 🧪 Q2 – Training + Hyperparameter Tuning with W&B Sweeps (15 Marks)

We trained the CNN model using:
- 80% of `train/` for training
- 20% of `train/` for validation  
*(using stratified split to preserve class balance)*

We used **Weights & Biases (W&B) Sweeps** to find the best configuration by exploring:
- `num_filters`: 32, 64
- `activation`: ReLU, GELU, SiLU
- `num_dense_neurons`: 128, 256
- `dropout`: 0.0, 0.2, 0.3
- `batch_norm`: True, False
- `lr`: 0.001, 0.0005

🔁 Strategy:
- Used **Bayesian Optimization** to reduce total runs  
- Tracked val accuracy using W&B dashboards

📊 Required plots (logged via wandb):
- Accuracy vs. Run (Created)
- Parallel Coordinates Plot
- Correlation Summary Table

---

## 📈 Q3 – Observations from Sweep Results (15 Marks)

Based on our W&B visualizations and CSV export from sweep logs, we observed:

✔️ **Batch Normalization** gives ~9% boost in accuracy on average  
✔️ Starting with **32 filters and doubling later** works better than starting with 64/128  
✔️ **ReLU** consistently outperforms GELU and SiLU in this shallow setting  
✔️ Dropout around **0.2–0.3** is stabilizing, but heavy dropout is not always best with BN  
✔️ **1e-3 learning rate** gave best results; lower rates underfit in our experiments  
✔️ Wider dense layers (256 vs 128) gave marginal boost (<1%)  

---

## 🧪 Q4 – Evaluation on Test Set (5 Marks)

Using the **best model checkpoint** from our sweep, we evaluated performance on the **unseen test set** (`val/` folder).

✅ Logged to W&B:
- ✅ Test accuracy  
- ✅ 10×3 image grid of predictions (green = correct, red = wrong)  
- ✅ 8×8 grid of first-layer filters  
- ✅ 10×1 Guided Backprop visualizations

📌 Notes:
- Model was never trained or tuned on test set  
- Guided backprop used Captum and predicted class as the target for visualization

---

## 🛠️ Environment & Tools

- PyTorch + Torchvision
- W&B for logging and sweeps
- Captum for interpretability
- Python 3.10+
- Ran on Kaggle GPU Notebook

---

## 🔗 Report

