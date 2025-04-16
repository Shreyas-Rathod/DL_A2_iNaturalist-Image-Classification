import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import wandb

# 0. To supress message in cell output
os.environ["WANDB_SILENT"] = "true"  
os.environ["WANDB_CONSOLE"] = "off"

# 1. Set your dataset paths (adjust if your path differs)
train_data_dir = "/kaggle/input/dl-a2-dataset/inaturalist_12K/train"
val_data_dir   = "/kaggle/input/dl-a2-dataset/inaturalist_12K/val"

# 2. Define transforms
#    (You can adjust image size or augmentations as desired)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),  
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 3. Create Datasets and DataLoaders
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(root=val_data_dir,   transform=val_transform)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)

# 4. Define a Configurable CNN Model
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
        We double filters each time, starting from 'num_filters'.
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
        
        # After 5 max-pools on a 64x64 image, the spatial size is 64/(2^5)=2, so feature map is (f * 2 * 2)
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

# 5. Define Training Function for wandb Sweep
def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Select activation function from config
        if config.activation == "ReLU":
            act_fn = nn.ReLU
        elif config.activation == "GELU":
            act_fn = nn.GELU
        elif config.activation == "SiLU":
            act_fn = nn.SiLU
        else:
            act_fn = nn.ReLU  # default fallback
        
        # Initialize model
        model = CNNModel(
            num_filters      = config.num_filters,
            kernel_size      = config.kernel_size,
            num_dense_neurons= config.num_dense_neurons,
            activation_fn    = act_fn,
            dropout          = config.dropout,
            batch_norm       = config.batch_norm
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        
        # Training Loop
        for epoch in range(config.epochs):
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
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / total
            train_acc  = correct / total
            
            # Validation
            model.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()
            
            val_loss /= total_val
            val_acc = correct_val / total_val
            
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            
            print(f"Epoch {epoch+1}/{config.epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save model checkpoint at the end of training
        torch.save(model.state_dict(), "model.pth")
        wandb.save("model.pth")

# 6. Define wandb Sweep Configuration
#    (You can customize hyperparameter ranges and search method)
sweep_config = {
    "method": "bayes",  # 'grid', 'random', or 'bayes'
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {
            "value": 20  # or choose a range if you want to sweep over epochs too
        },
        "lr": {
            "values": [0.001, 0.0005]
        },
        "num_filters": {
            "values": [32, 64, 128, 256, 512]
        },
        "kernel_size": {
            "value": 3
        },
        "num_dense_neurons": {
            "values": [128, 256]
        },
        "activation": {
            "values": ["ReLU", "GELU", "SiLU"]
        },
        "dropout": {
            "values": [0.0, 0.2, 0.3]
        },
        "batch_norm": {
            "values": [True, False]
        }
    }
}

# 7. Initialize the Sweep
sweep_id = wandb.sweep(sweep_config, project="CS24M046_DA6401_A2_t1")

# 8. Run the Sweep Agent (Adjust count as desired)
wandb.agent(sweep_id, function=train_model, count=50)
