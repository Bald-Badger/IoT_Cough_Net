import torch
import torch.nn as nn
import torch.nn.functional as F

class CoughNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 8)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(32, 16, 8)
        self.fc1 = nn.Linear(800, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 1)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = torch.flatten(x)
        x = torch.softmax(x, dim=0)
        return x
