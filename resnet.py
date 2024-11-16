import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 500  # Number of output classes


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Load a pre-trained ResNet model (e.g., ResNet-50)
        self.resnet = models.resnet50(pretrained=True)
        # keep resnet features frozen
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the fully connected (fc) layer to match the number of classes
        # ResNet's fc input features can be accessed as self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, nclasses)

    def forward(self, x):
        # Forward pass through ResNet
        return self.resnet(x)