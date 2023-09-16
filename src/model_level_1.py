import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MANUAL_SEED

torch.manual_seed(MANUAL_SEED)


class Net(nn.Module):
    def __init__(self, input_units, output_units):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_units, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_units),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)
