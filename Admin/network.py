import dataset_loader
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

    self.fc1 = nn.Linear(in_features=12*13*13, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=2)

  def forward(self, t):
    t = self.conv1(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    t = F.relu(self.conv2(t))
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    t = t.reshape(-1, 12 * 13 * 13)
    t = F.relu(self.fc1(t))
    t = F.relu(self.fc2(t))
    t = self.out(t)

    return t

# conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
# conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
# fc1 = nn.Linear(in_features=12*13*13, out_features=120)

# train_set, train_labels, test_set, test_labels, val_set, val_labels, BATCH_SIZE = dataset_loader.load_dataset()
# images = train_set[0]
# images = np.array(images).astype(np.int32)
# images = torch.Tensor(images)

# t = images
# t = conv1(t)
# t = F.relu(t)
# t = F.max_pool2d(t, kernel_size=2, stride=2)
# t = F.relu(conv2(t))
# t = F.max_pool2d(t, kernel_size=2, stride=2)

# print(t.shape)

# t = t.reshape(-1, 12 * 13 * 13)
# t = F.relu(fc1(t))

# print(t.shape)