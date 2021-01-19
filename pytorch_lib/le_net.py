# coding: utf-8

import torch.nn.functional as f
import torch.nn.modules as mod


class LeNet(mod.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input, 6 output, 5x5 convolution
        self.conv1 = mod.Conv2d(1, 6, 1)
        # 6 input, 16 output, 5x5 convolution
        self.conv2 = mod.Conv2d(6, 16, 5)
        # full connection
        self.fc1 = mod.Linear(16 * 5 * 5, 120)
        self.fc2 = mod.Linear(120, 84)
        self.fc3 = mod.Linear(84, 10)
        self.dropout = mod.Dropout(0.2)

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        x = f.max_pool2d(f.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = f.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
