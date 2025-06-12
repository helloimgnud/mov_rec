import numpy as np
import torch
from torch import nn, div, square, norm
from torch.nn import functional as F

class AutoRec(nn.Module):
    def __init__(self, d, k, dropout):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d, k),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(k, d)
        )

    def forward(self, r):
        return self.seq(r)

class UserFeaturesNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_train_users):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_train_users) 
        )

    def forward(self, x):
        weights = self.net(x)  
        weights = F.softmax(weights, dim=1)  
        return weights