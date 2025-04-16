import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10,
                 conv_filters=[32, 64, 128, 256, 512],
                 filter_size=3,
                 activation_fn=nn.ReLU,
                 dense_neurons=256):
        super(CustomCNN, self).__init__()
        
        self.conv_layers = nn.Sequential()
        current_in_channels = in_channels

        for i, out_channels in enumerate(conv_filters):
            self.conv_layers.add_module(f'conv{i+1}', nn.Conv2d(current_in_channels, out_channels, kernel_size=filter_size, padding=1))
            self.conv_layers.add_module(f'act{i+1}', activation_fn())
            self.conv_layers.add_module(f'pool{i+1}', nn.MaxPool2d(kernel_size=2))
            current_in_channels = out_channels

        self.flatten = nn.Flatten()

        # Assume input image 224x224, compute final size
        final_size = 224 // (2 ** 5)  # 5 pool layers halve it 5 times
        self.feature_dim = conv_filters[-1] * final_size * final_size

        self.fc1 = nn.Linear(self.feature_dim, dense_neurons)
        self.out = nn.Linear(dense_neurons, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.out(x)
        return x
