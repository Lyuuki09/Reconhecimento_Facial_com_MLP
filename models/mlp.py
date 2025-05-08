import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=64*64*3, num_classes=3, hidden_layers=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x) 