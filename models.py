import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleKeypointModel(nn.Module):
    """ Simple model to detect glue tube keypoints.
    """
    def __init__(self,logit_shape:int = 8, dropout_proba:float = 0.5):

      super(SimpleKeypointModel, self).__init__()
      #conv layers
      self.conv1 = nn.Conv2d(3, 32, 7,stride = 2)
      self.conv2 = nn.Conv2d(32, 64, 5,stride = 2)
      self.conv3 = nn.Conv2d(64, 128, 3,stride = 1)
      #pooling layer
      self.maxpool = nn.MaxPool2d(2, 2)
      #dropout
      self.dropout = nn.Dropout(p=dropout_proba)
      #fully connected layers
      self.fc1 = nn.Linear(128*14*14,256)
      self.fc2 = nn.Linear(256,logit_shape)

    def forward(self,x):

      x = F.relu(self.maxpool(self.conv1(x)))
      x = F.relu(self.maxpool(self.conv2(x)))
      x = F.relu(self.maxpool(self.conv3(x)))

      x = self.dropout(x.view(x.size(0), -1))
      x = self.dropout(self.fc1(x))
      x = self.fc2(x)
      return x
    
