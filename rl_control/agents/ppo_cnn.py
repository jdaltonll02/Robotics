import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten()
        )
    def forward(self, x):
        return self.cnn(x)

class PPOActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.feature_extractor = CNNFeatureExtractor(obs_shape[0])
        n_flatten = 3136  # Adjust based on input size
        self.actor = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(n_flatten, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.actor(features), self.critic(features)
