# Question 4 

!pip install -q captum
import os
import wandb
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import warnings
from captum.attr import GuidedBackprop

# ------------------ W&B Setup ------------------
wandb.login()
wandb.init(project="CS24M046_DA6401_A2_t1", name="final-eval", job_type="evaluation")

# ------------------ Dataset Setup ------------------
TEST_DIR = "/kaggle/input/dl-a2-dataset/inaturalist_12K/val"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_tf = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
test_set = datasets.ImageFolder(TEST_DIR, transform=test_tf)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
class_names = test_set.classes

# ------------------ CNN Model ------------------
class CNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        activation_fn = torch.nn.ReLU
        num_filters = 32
        kernel_size = 3
        dropout = 0.0
        num_dense_neurons = 256
        batch_norm = True

        in_channels = 3
        filters = [num_filters, num_filters*2, num_filters*4, num_filters*8, num_filters*16]
        self.conv_blocks = torch.nn.ModuleList()

        for f in filters:
            block = [torch.nn.Conv2d(in_channels, f, kernel_size=kernel_size, padding=1),
                     activation_fn()]
            if batch_norm:
                block.append(torch.nn.BatchNorm2d(f))
            block.append(torch.nn.MaxPool2d(2))
            if dropout > 0:
                block.append(torch.nn.Dropout(dropout))
            self.conv_blocks.append(torch.nn.Sequential(*block))
            in_channels = f

        self.flattened_size = filters[-1] * 2 * 2
        self.fc1 = torch.nn.Linear(self.flattened_size, num_dense_neurons)
        self.fc2 = torch.nn.Linear(num_dense_neurons, 10)
        self.activation = activation_fn()

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        return self.fc2(x)

# ------------------ Load Model ------------------
CHECKPOINT_PATH = "/kaggle/input/modelinfo/model.pth"  # <-- update if needed
model = CNNModel().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# ------------------ Test Accuracy ------------------
correct, total = 0, 0
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs).argmax(1)
        total += lbls.size(0)
        correct += (preds == lbls).sum().item()

test_acc = correct / total
print(f"\n🟢 Test accuracy: {test_acc*100:.2f}%")
wandb.log({"test_accuracy": test_acc})

# ------------------ 10x3 Prediction Grid ------------------
rows, cols = 10, 3
fig1, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
model.cpu()
for ax in axes.ravel():
    idx = np.random.randint(0, len(test_set))
    img, lbl = test_set[idx]
    pred = model(img.unsqueeze(0)).argmax(1).item()
    ax.imshow(img.permute(1,2,0))
    ax.set_title(f"P:{class_names[pred]}\nT:{class_names[lbl]}", fontsize=8, color="g" if pred==lbl else "r")
    ax.axis('off')
plt.tight_layout()
wandb.log({"predictions_grid": wandb.Image(fig1)})
plt.close(fig1)

# ------------------ First Layer Filters ------------------
filters = model.conv_blocks[0][0].weight.data.clone()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
grid = torchvision.utils.make_grid(filters, nrow=8, padding=1)
fig2 = plt.figure(figsize=(8,8))
plt.title("First-layer Filters")
plt.imshow(grid.permute(1,2,0))
plt.axis('off')
wandb.log({"first_layer_filters": wandb.Image(fig2)})
plt.close(fig2)

# ------------------ Guided Backprop (10 images) ------------------
warnings.filterwarnings("ignore", category=UserWarning)
gbp = GuidedBackprop(model.to(device))
model.eval()
imgs, labels = next(iter(test_loader))
imgs = imgs[:10].to(device)
with torch.no_grad():
    preds = model(imgs).argmax(1)

attributions = []
for i in range(10):
    attr = gbp.attribute(imgs[i].unsqueeze(0), target=preds[i].item())
    attributions.append(attr.cpu().squeeze().numpy())

fig3, axs = plt.subplots(1, 10, figsize=(20, 2))
for i in range(10):
    attr = attributions[i]
    attr = np.moveaxis(attr, 0, -1)
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
    axs[i].imshow(attr)
    axs[i].axis('off')
plt.suptitle("Guided Backprop – 10 Test Images", fontsize=12)
plt.tight_layout()
wandb.log({"guided_backprop_grid": wandb.Image(fig3)})
plt.close(fig3)

# ------------------ Finish ------------------
wandb.finish()
