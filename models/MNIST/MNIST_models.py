import torch
from torch import nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels= 1,
                out_channels= 16,
                kernel_size= 5,
                stride= 1,
                padding= 2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten
        x = x.reshape(x.size(0), -1)
        output =self.fc(x)

        return output

class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_features=784, out_features=512),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                 nn.ReLU(), nn.Dropout(0.1))
        self.fc = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # Flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc(x)

        return x

class MNIST_MCLR(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(MNIST_MCLR, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Flatten
        x = x.view(x.size(0), -1)
        return self.fc(x)
