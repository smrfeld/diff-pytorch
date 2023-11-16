import torch
from typing import Tuple, List

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(torch.nn.Module):

    def __init__(self, num_channels_input: int, num_channels_output: int):
        super(ResidualBlock, self).__init__()
        self.num_channels_input = num_channels_input
        self.num_channels_output = num_channels_output

        # Change num channels from num_channels_input to num_channels_output
        self.conv_change_dim = torch.nn.Conv2d(in_channels=num_channels_input, out_channels=num_channels_output, kernel_size=1)

        #  x = layers.BatchNormalization(center=False, scale=False)(x)
        # Center=False and scale=False => don't apply scale or shift to the normalized output
        # i.e. the batch norm just takes off the mean and divides by std, but does not scale/shift result
        # Equivalent to affine=False
        self.batch_norm = torch.nn.BatchNorm2d(num_features=num_channels_input, affine=False)

        self.conv_1 = torch.nn.Conv2d(
            in_channels=num_channels_input, out_channels=num_channels_output, kernel_size=3, padding="same"
            )
        self.conv_2 = torch.nn.Conv2d(
            in_channels=num_channels_output, out_channels=num_channels_output, kernel_size=3, padding="same"
            )
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Cast into correct shape = width
        if self.num_channels_input != self.num_channels_output:
            residual = self.conv_change_dim(x)
        else:
            residual = x

        x = self.batch_norm(x)
        x = self.conv_1(x)
        x = self.swish(x)
        x = self.conv_2(x)
        return x + residual


class DownBlock(torch.nn.Module):

    def __init__(self, num_channels_input: int, num_channels_output: int, block_depth: int):
        """Input shape: (batch_size, num_features, height, width)
        Output shape: (batch_size, num_features, height // 2, width // 2)
        """
        super(DownBlock, self).__init__()
        self.block_depth = block_depth
        self.rbs = []
        for i in range(block_depth):
            num_channels_in = num_channels_input if i == 0 else num_channels_output
            rb = ResidualBlock(num_channels_in, num_channels_output)
            self.rbs.append(rb)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, x: Tuple[torch.Tensor, List]) -> torch.Tensor:
        y, skips = x
        for rb in self.rbs:
            y = rb(y)
            skips.append(y)
        y = self.avg_pool(y)
        return y


class UpBlock(torch.nn.Module):

    def __init__(self, num_channels_input: int, num_channels_output: int, block_depth: int):
        super(UpBlock, self).__init__()
        self.block_depth = block_depth
        self.rbs = []
        for i in range(block_depth):
            num_channels_in = num_channels_input if i == 0 else num_channels_output
            # Multiply input by 2 => add skip connection
            num_channels_in *= 2
            rb = ResidualBlock(num_channels_in, num_channels_output)
            self.rbs.append(rb)
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: Tuple[torch.Tensor, List]) -> torch.Tensor:
        y, skips = x
        y = self.up(y)
        for rb in self.rbs:
            # Catenate along features
            y = torch.cat([y, skips.pop()], dim=1)
            y = rb(y)
        return y

