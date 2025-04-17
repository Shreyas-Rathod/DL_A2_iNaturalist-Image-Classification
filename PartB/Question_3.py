import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# =============================
# Config: W&B will sweep over this
# =============================
sweep_config = {
    "method": "grid",
    "parameters": {
        "strategy": {
            "values": ["freeze_all_but_fc", "freeze_until_layer4", "train_all"]
        },
        "epochs": {"value": 5},
        "lr": {"value": 0.0005}
    }
}

# =============================
# Data Loaders
# =============================
def get_data(batch_size=32):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder("/kaggle/input/dl-a2-dataset/inaturalist_12K/train", tf)
    val_ds   = datasets.ImageFolder("/kaggle/input/dl-a2-dataset/inaturalist_12K/val", tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# =============================
# Apply Freezing Strategy
# =============================
def apply_strategy(model, strategy):
    for param in model.parameters():
        param.requires_grad = False  # freeze everything first

    if strategy == "freeze_all_but_fc":
        for param in model.fc.parameters():
            param.requires_grad = True

    elif strategy == "freeze_until_layer4":
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    elif strategy == "train_all":
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# =============================
# Sweep Training Function
# =============================
def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config
        strategy = config.strategy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loader = get_data()

        # Load model + replace head
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.to(device)

        # Apply freezing
        apply_strategy(model, strategy)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(config.epochs):
            model.train()
            total, correct, loss_total = 0, 0, 0.0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                loss_total += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

            train_acc = correct / total
            train_loss = loss_total / total

            # Validation
            model.eval()
            total, correct, loss_total = 0, 0, 0.0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    loss_total += loss.item() * x.size(0)
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)

            val_acc = correct / total
            val_loss = loss_total / total

            # Log metrics
            wandb.log({
                "epoch": epoch + 1,
                "strategy": strategy,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

# =============================
# Launch Sweep
# =============================
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="CS24M046_DA6401_A2_t1")
    wandb.agent(sweep_id, function=train_model, count=3)  # one for each strategy
