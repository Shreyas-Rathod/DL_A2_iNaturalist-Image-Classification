import torch
import torch.nn as nn
import torch.optim as optim

class CNNModel(nn.Module):
    def __init__(self, num_filters=32, kernel_size=3, num_dense_neurons=128):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv2d(num_filters*2, num_filters*4, kernel_size=kernel_size, padding=1)
        self.conv4 = nn.Conv2d(num_filters*4, num_filters*8, kernel_size=kernel_size, padding=1)
        self.conv5 = nn.Conv2d(num_filters*8, num_filters*16, kernel_size=kernel_size, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)  # Max-pooling layer with 2x2 window
        
        self.fc1 = nn.Linear(num_filters*16*8*8, num_dense_neurons)  # Adjust input features after flattening
        self.fc2 = nn.Linear(num_dense_neurons, 10)  # Output layer with 10 classes

        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer
        
        return x
