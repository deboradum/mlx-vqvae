import mlx.nn as nn

from models.residual import ResidualStack


class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, num_res_layers):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, num_res_layers),
        )

    def __call__(self, x):
        return self.layers(x)
