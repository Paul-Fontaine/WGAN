import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, checkpoint_path: str = None):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 128, 7, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )
        if checkpoint_path:
            self.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))

    def forward(self, z):
        return self.model(z)


class Critic(nn.Module):
    def __init__(self, checkpoint_path: str = None):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )
        if checkpoint_path:
            self.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))

    def forward(self, x):
        return self.model(x)
