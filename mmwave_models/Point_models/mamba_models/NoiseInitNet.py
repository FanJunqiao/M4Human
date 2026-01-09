import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np

import math


class NoiseInitNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=4):
        super().__init__()
        layers = []

        # First layer
        layers.append(nn.LayerNorm(input_dim))
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Middle layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Final layer maps to 2 * output_dim (mean and logvar)
        layers.append(nn.Linear(hidden_dim, 2 * output_dim))

        self.encoder = nn.Sequential(*layers)
        # Initialize weights of self.encoder to zero
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):  # x: (K, B, N, C)
        K, B, N, C = x.shape
        x = x.view(K * B, N * C)
        stats = self.encoder(x)
        mu, logvar = stats.chunk(2, dim=-1)
        mu = mu.view(K, B, N, C)
        logvar = logvar.view(K, B, N, C)
        return mu, logvar
    
    def sample(self, x): 
        mu, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(x)
        print("mu", mu.mean(), "std", std.mean())
        return mu + eps * std
    
    def compute_loss(self, pred_motion, gt_motion, lamb=1e-2):
        # Diversity loss: maximize the pairwise distance among K hypotheses
        K, B, N, C = pred_motion.shape
        pred_motion_flat = pred_motion.view(K, -1)  # Flatten to (K, B * N * C) [...,[0],:]
        diversity_loss = 0
        for i in range(K):
            for j in range(i + 1, K):
                diversity_loss += torch.norm(pred_motion_flat[i] - pred_motion_flat[j], p=2)
        diversity_loss /= (K * (K - 1) / 2)  # Normalize by the number of pairs

        # Min error loss: minimize the error between the closest hypothesis and gt_motion
        error = torch.norm(pred_motion - gt_motion, dim=(2, 3))  # Compute error for each hypothesis
        min_error_loss = torch.min(error, dim=0)[0].mean()  # Take the minimum error across K hypotheses and average
        # plot_motion = torch.cat([pred_motion, gt_motion], dim=-1).detach()
        # from plot import plot_frequency_diagram
        # plot_frequency_diagram(plot_motion[0,0])

        print(f"min_error_loss: {min_error_loss.item()}, diversity_loss: {diversity_loss.item()}")

        # Combine losses
        loss = min_error_loss - lamb*diversity_loss
        return loss * 0.1
