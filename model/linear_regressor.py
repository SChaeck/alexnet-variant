import torch.nn as nn

class LinearRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.regressor(x)