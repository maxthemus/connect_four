import torch
from torch import nn

class ConnectFourModel(nn.Module):
    def __init__(self):
        super(ConnectFourModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 5, 256)
        self.fc2 = nn.Linear(256, 7)  # 7 possible columns to drop a piece

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x