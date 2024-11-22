import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 500  # Number of output classes


class Dinov2CLF(nn.Module):
    def __init__(self, frozen: bool = True):
        super(Dinov2CLF, self).__init__()
        # Load the pretrained weights of dinov2
        self.dinov2_clf = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        # Freeze the weights of the model
        if frozen:
            for param in self.dinov2_clf.parameters():
                param.requires_grad = False
        # Replace the fully connected (fc) layer to match the number of classes
        # DINO's fc input features can be accessed as self.resnet.head.in_features
        in_features = self.dinov2_clf.norm.normalized_shape[0]
        linear_clf = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, nclasses)
        )
        self.dinov2_clf.head = linear_clf
        # self.dinov2_clf.head = nn.Linear(in_features, nclasses)


    def forward(self, x):
        # Forward pass through ResNet
        return self.dinov2_clf(x)