import torch
import torch.nn as nn

class AlexNetWithResidual(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        )
        self.features2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
        )
        self.features3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.features5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.embedding = nn.Sequential( # 임베딩을 확인하기 위해 분리
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.residualConv1 = nn.Conv2d(64, 192, kernel_size=4, stride=2, padding=1)
        self.residualConv2 = nn.Conv2d(192, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x = self.features1(x)
        x = self.features2(x)
        x = x + self.residualConv1(skip)
        skip = x = self.features3(x)
        x = self.features4(x)
        x = x + self.residualConv2(skip)
        x = self.features5(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        embedding = x
        x = self.classifier(x)
        return x, embedding            