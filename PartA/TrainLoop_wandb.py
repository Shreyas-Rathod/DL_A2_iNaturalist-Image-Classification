import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision
from torchvision import datasets, transforms
import wandb

# ===========================================
# 1. Dataset Paths 
# ===========================================
# Provided folders:
# - 'train': Original training data (to be split into train and validation)
# - 'val': Test data
train_data_dir = "/kaggle/input/dl-a2-dataset/inaturalist_12K/train"
test_data_dir  = "/kaggle/input/dl-a2-dataset/inaturalist_12K/val"  # This is the test set

# ===========================================
# 2. Define Data Transforms
# ===========================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# Use a simpler transform for validation extracted from training
val_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])
# For the test set, we use the same as val_transform
test_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

# ===========================================
# 3. Load the Training Dataset and Perform Stratified Split
# ===========================================
# Load entire training data
full_train_dataset = datasets.ImageFolder(root=train_data_dir, transform=train_transform)

# Use targets for stratified splitting
targets = np.array(full_train_dataset.targets)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in sss.split(np.zeros(len(targets)), targets):
    train_subset = Subset(full_train_dataset, train_idx)
    # For validation, use the same files but override the transform to val_transform
    train_val_dataset = datasets.ImageFolder(root=train_data_dir, transform=val_transform)
    val_subset = Subset(train_val_dataset, val_idx)

# Create DataLoaders for training and validation splits
batch_size = 32
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

# ===========================================
# 4. Load the Test Dataset (from provided 'val' folder)
# ===========================================
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ===========================================
# 5. Define a Configurable CNN Model
# ===========================================
class CNNModel(nn.Module):
    def __init__(self, 
                 num_filters=32, 
                 kernel_size=3, 
                 num_dense_neurons=128,
                 activation_fn=nn.ReLU, 
                 dropout=0.0, 
                 batch_norm=False):
        """
        5 convolutional blocks:
          - Each block: Conv -> Activation -> (Optional BN) -> MaxPool -> (Optional Dropout)
        Filters double each block starting from 'num_filters'.
        """
        super(CNNModel, self).__init__()
        in_channels = 3
        filters = [num_filters, num_filters*2, num_filters*4, num_filters*8, num_filters*16]
        self.conv_blocks = nn.ModuleList()
        for f in filters:
            block = []
            block.append(nn.Conv2d(in_channels, f, kernel_size=kernel_size, padding=1))
            block.append(activation_fn())
            if batch_norm:
                block.append(nn.BatchNorm2d(f))
            block.append(nn.MaxPool2d(2))
            if dropout > 0:
                block.append(nn.Dropout(dropout))
            self.conv_blocks.append(nn.Sequential(*block))
            in_channels = f
        
        # After 5 poolings on a 64x64 input, spatial dimensions become 64/(2^5)=2.
        self.flattened_size = filters[-1] * 2 * 2
        self.fc1 = nn.Linear(self.flattened_size, num_dense_neurons)
        self.fc2 = nn.Linear(num_dense_neurons, 10)  # 10 classes
        self.activation = activation_fn()

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# ===========================================
# 6. Define Training Function for wandb Sweep
# ===========================================
def train_model(config=None):
    # Reduce wandb verbosity in output
    import os
    os.environ["WANDB_SILENT"] = "true"  
    os.environ["WANDB_CONSOLE"] = "off"
    
    with wandb.init(
        project="CS24M046_DA6401_A2_t1",
        config=config,
        settings=wandb.Settings(console="off", silent=True)
    ):
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Choose activation function based on config
        if config.activation == "ReLU":
            act_fn = nn.ReLU
        elif config.activation == "GELU":
            act_fn = nn.GELU
        elif config.activation == "SiLU":
            act_fn = nn.SiLU
        else:
            act_fn = nn.ReLU  # default fallback
        
        model = CNNModel(
            num_filters       = config.num_filters,
            kernel_size       = config.kernel_size,
            num_dense_neurons = config.num_dense_neurons,
            activation_fn     = act_fn,
            dropout           = config.dropout,
            batch_norm        = config.batch_norm
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        
        epochs = config.epochs
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
            
            train_loss = running_loss / total
            train_acc  = correct / total
            
            # Evaluate on validation subset extracted from training data
            model.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, preds = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += preds.eq(labels).sum().item()
            
            val_loss /= total_val
            val_acc = correct_val / total_val
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the model checkpoint
        torch.save(model.state_dict(), "model.pth")
        wandb.save("model.pth")

# ===========================================
# 7. Define wandb Sweep Configuration
# ===========================================
sweep_config = {
    "method": "bayes",  # 'grid', 'random', or 'bayes'
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {"value": 20},  # Fixed for quick experiments
        "lr": {"values": [0.001, 0.0005]},
        "num_filters": {"values": [32, 64, 128, 256, 512]},
        "kernel_size": {"value": 3},
        "num_dense_neurons": {"values": [128, 256]},
        "activation": {"values": ["ReLU", "GELU", "SiLU"]},
        "dropout": {"values": [0.0, 0.2, 0.3]},
        "batch_norm": {"values": [True, False]}
    }
}

# ===========================================
# 8. Initialize and Run the Sweep Agent
# ===========================================
sweep_id = wandb.sweep(sweep_config, project="CS24M046_DA6401_A2_t1")
print("Sweep ID:", sweep_id)
wandb.agent(sweep_id, function=train_model, count=40)
