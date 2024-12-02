import torch
import torch.nn as nn
from model import AlexNet, LinearRegressor

class AlexNetWithClassifier(nn.Module):

    def __init__(self, alexnet: AlexNet, classifier: LinearRegressor) -> None:
        super().__init__()
        self.alexnet = alexnet
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alexnet.features(x)
        x = self.alexnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.alexnet.embedding(x) 
        embedding = x
            
        x = self.classifier(x)
        return x, embedding