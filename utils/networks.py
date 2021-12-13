#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torchvision.models as models
from torch import nn


# configuration module
# ------
import config


# ----------------
# networks
# ----------------   

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # regular resnet 18 as feature extractor
        resnet18 = models.resnet18(pretrained=False)
        # setting the final layer followed by an MLP to be the representation
        # network (encoder)
        setattr(resnet18, 'fc', nn.Linear(512, config.HIDDEN_DIM))
        self.encoder = nn.Sequential(resnet18, nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(), nn.Linear(config.HIDDEN_DIM, config.FEATURE_DIM, bias=False))
        # a linear layer as projection network
        self.projector = MLPHead(config.FEATURE_DIM, config.HIDDEN_DIM, config.FEATURE_DIM)

    def forward(self, x):
        """
        x: image tensor of (B x 3 x 64 x 64)
        return: representation tensor h (B x FEATURE_DIM), projection tensor z
        (B x HIDDEN_DIM) that should be used by the loss function.
        """
        representation = self.encoder(x)
        projection = self.projector(representation)
        return representation, projection


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)
# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment