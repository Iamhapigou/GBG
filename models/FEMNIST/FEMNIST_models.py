import torch
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel=1, out_channel=16, stride=1):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels= in_channel,
                out_channels= out_channel,
                kernel_size= 3,
                stride= stride,
                padding= 1
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(
                in_channels= out_channel,
                out_channels= out_channel,
                kernel_size= 3,
                stride= 1,
                padding= 1
            ),
            nn.BatchNorm2d(out_channel)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(
            in_channels= in_channel,
            out_channels = out_channel,
            kernel_size = 1,
            stride= stride)
        )

    def forward(self, x):
        out_1 = self.layer(x)
        out_2 = self.shortcut(x)
        out = out_1 + out_2
        out = F.relu(out)

        return out

class FEMNIST_ResNet(nn.Module):
    def make_layer(self, in_channel, out_channel, stride, num_block=1):
        layer = []
        channel = in_channel
        in_stride = stride
        for i in range(num_block):
            if i == 0:
                pass
            else:
                in_stride = 1
            layer.append(ResBlock(channel,
                               out_channel,
                               in_stride))
            channel = out_channel

        return nn.Sequential(*layer)

    def __init__(self):
        super(FEMNIST_ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(16, 128, 2, 2)

        self.fc = nn.Linear(128 * 7 * 7, 62)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)

        out = F.avg_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)

        output = self.fc(out)

        return output

class FEMNIST_CNN(nn.Module):
    def __init__(self):
        super(FEMNIST_CNN, self).__init__()
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

        self.fc = nn.Linear(32*7*7, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten
        x = x.reshape(x.size(0), -1)
        output =self.fc(x)

        return output

class FEMNIST_CNN2(nn.Module):
    def __init__(self):
        super(FEMNIST_CNN2, self).__init__()
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

        self.fc1 = nn.Linear(32*7*7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten
        x = x.reshape(x.size(0), -1)
        x =self.fc1(x)
        x = self.fc2(x)
        output = self.fc(x)

        return output

class FEMNIST_MLP(nn.Module):
    def __init__(self):
        super(FEMNIST_MLP, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_features=784, out_features=512),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                 nn.ReLU(), nn.Dropout(0.1))
        self.fc3 = nn.Linear(in_features=256, out_features=62)

    def forward(self, x):
        # Flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class FEMNIST_MCLR(nn.Module):
    def __init__(self, input_dim=784, output_dim=62):
        super(FEMNIST_MCLR, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Flatten
        x = x.view(x.size(0), -1)
        return self.linear(x)

