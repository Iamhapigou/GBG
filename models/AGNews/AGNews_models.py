import torch
import torch.nn.functional as F
from torch import nn

class TextCNN(nn.Module):
    def __init__(self, hidden_dim=128, num_channels=100, kernel_size=[3, 4, 5], max_len=200, dropout=0.3,
                 padding_idx=0, vocab_size=32000, num_classes=4):
        super(TextCNN, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(dropout)

        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels * len(kernel_size), num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text).permute(0, 2, 1)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)
        out = F.log_softmax(out, dim=1)

        return out

class LSTMNet(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, bidirectional=True, dropout=0.2,
                 padding_idx=0, vocab_size=32000, num_classes=4):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        dims = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, text_lengths = x
        else:
            text, text_lengths = x, [x.shape[1] for _ in range(x.shape[0])]

        embedded = self.embedding(text)
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # unpack sequence
        if self.lstm.bidirectional:
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden_cat = hidden[-1]

        out = self.dropout(hidden_cat)
        out = self.fc(out)

        return out

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=32000, embed_dim=128, num_heads=4, num_layers=3, num_classes=4, dropout=0.1):
        super(TinyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, text_lengths = x
        else:
            text, text_lengths = x, [x.shape[1] for _ in range(x.shape[0])]

        x = self.embedding(text).permute(1, 0, 2)
        enc_out = self.encoder(x)
        enc_out = enc_out.mean(dim=0)
        return self.fc(enc_out)

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample
        if downsample or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class ResNetText(nn.Module):
    def __init__(self, vocab_size=32000, embed_dim=128, num_classes=4, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.layer1 = BasicBlock1D(embed_dim, 128)
        self.layer2 = BasicBlock1D(128, 64)
        self.dropout = nn.Dropout(0.3)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, text_lengths = x
        else:
            text, text_lengths = x, [x.shape[1] for _ in range(x.shape[0])]

        x = self.embedding(text)          # [B, L, D]
        x = x.permute(0, 2, 1)         # [B, D, L]
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.global_pool(x).squeeze(-1)  # [B, 128]
        x = self.dropout(x)
        return self.fc(x)


