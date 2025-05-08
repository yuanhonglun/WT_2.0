import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),  # 增加Dropout
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=64):
        super(SiameseNetwork, self).__init__()

        self.autoencoder = Autoencoder(input_dim, latent_dim)

        self.shared_network = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, stride=1, padding=1),  # 减小kernel
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.4),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2 * (latent_dim // 4), 32),  # 降低维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 8)  # 进一步降低输出维度
        )

        self.fc = nn.Sequential(
            nn.Linear(8 + 1, 4),  # 更简单的全连接层
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, left, right, left_rt, right_rt):
        left_latent = self.autoencoder(left).unsqueeze(1)
        right_latent = self.autoencoder(right).unsqueeze(1)

        left_emb = self.shared_network(left_latent)
        right_emb = self.shared_network(right_latent)

        distance = torch.abs(left_emb - right_emb)
        rt_diff = torch.abs(left_rt - right_rt).unsqueeze(1)

        combined = torch.cat((distance, rt_diff), dim=1)
        return self.fc(combined)

