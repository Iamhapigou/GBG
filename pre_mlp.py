import torch
import torch.nn.functional as F
import torch.nn as nn

class pre_MLP(nn.Module):
    def __init__(self, input_dim = 784, hidden_dim=256, output_dim = 10):
        super(pre_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(F.relu(x))

        return x

class cifar100_MLP(nn.Module):
    def __init__(self, input_dim = 784, hidden_dim=100, output_dim = 100):
        super(cifar100_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #flatten
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(F.relu(x))

        return x

class AGNewsMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x[0])
        mean_emb = emb.mean(dim=1)
        out = self.fc(mean_emb)
        return out
