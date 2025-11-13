import torch
from torch import nn
import torch.nn.functional as F

class ConnectFourModel(nn.Module):
    def __init__(self):
        super(ConnectFourModel, self).__init__()
        self.fc1 = nn.Linear(42, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 7)  # Output: Q-value for each column
        
        self.dropout = nn.Dropout(0.2)  # Dropout for regularization

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x