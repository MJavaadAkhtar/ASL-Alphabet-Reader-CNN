from dataPreparation import *

NumberOfUniqueClasses = 29 # !IMPORTANT! change this base number of classes


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=4,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4,
                               out_channels=8,
                               kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=3,
                               padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=3,
                               padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, NumberOfUniqueClasses)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.bn3(x)
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.bn4(x)
        x = x.view(-1, 32 * 12 * 12)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)