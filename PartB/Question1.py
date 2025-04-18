!pip install -q wandb
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# =============================
# 1. Init W&B run
# =============================
wandb.login()
wandb.init(project="CS24M046_DA6401_A2_t1", name="resnet50-finetune", job_type="finetuning")

# =============================
# 2. Transforms (224x224 + ImageNet stats)
# =============================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =============================
# 3. Load Dataset
# =============================
train_dir = "/kaggle/input/dl-a2-dataset/inaturalist_12K/train"
val_dir   = "/kaggle/input/dl-a2-dataset/inaturalist_12K/val"

train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
val_ds   = datasets.ImageFolder(val_dir, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
class_names = train_ds.classes

# =============================
# 4. Load Pretrained ResNet-50
# =============================
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # replace final layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =============================
# 5. Loss + Optimizer
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# =============================
# 6. Training + Validation Loop
# =============================
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_train, correct_train, train_loss = 0, 0, 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)
        train_loss += loss.item() * imgs.size(0)

    train_acc = correct_train / total_train
    train_loss = train_loss / total_train

    # -----------------------
    # Validation
    # -----------------------
    model.eval()
    correct_val, total_val, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            val_loss += loss.item() * imgs.size(0)

    val_acc = correct_val / total_val
    val_loss = val_loss / total_val

    # -----------------------
    # W&B Logging
    # -----------------------
    wandb.log({
        "epoch": epoch+1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# =============================
# 7. Sample Predictions Logging
# =============================
model.eval()
images, labels = next(iter(val_loader))
images, labels = images[:16].to(device), labels[:16].to(device)
with torch.no_grad():
    preds = model(images).argmax(1)

# Denormalize
def denorm(img):
    img = img.cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    return np.clip((img * std + mean), 0, 1)

# Log 16 images
log_imgs = []
for i in range(16):
    img = denorm(images[i])
    pred = class_names[preds[i]]
    true = class_names[labels[i]]
    caption = f"Pred: {pred} | True: {true}"
    log_imgs.append(wandb.Image(img, caption=caption))

wandb.log({"sample_predictions": log_imgs})
wandb.finish()
