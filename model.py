from torch import nn
import torch.nn.functional as F

class ShapesCNN(nn.Module):
    def __init__(self):
        super(ShapesCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 50 * 50, 512)
        self.fc2 = nn.Linear(512, 9)

        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.reshape(x.size(0), -1)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x