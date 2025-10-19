import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CIFAR100_MLP(nn.Module):
    def __init__(self, input_dim=3*32*32, hidden_dim=512, num_classes=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten [B, 3, 32, 32] → [B, 3072]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR100_CNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32→16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16→8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8→4
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CIFAR100_ResNet18(nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super().__init__()

        weights = None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1
        m = models.resnet18(weights=weights)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #FedKD,PFL_DA
        #m.fc = nn.Identity()
        m.fc = nn.Linear(512,100)
        self.backbone = m

    def forward(self, x):
        return self.backbone(x)

class CIFAR100_EfficientNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class CIFAR100_MobileNetV2(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.backbone.features[0][0].stride = (1, 1)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.backbone(x)
