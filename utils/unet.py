from .sinusoidal import sinusoidal_embedding

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
        rbs = []
        for i in range(block_depth):
            num_channels_in = num_channels_input if i == 0 else num_channels_output
            rb = ResidualBlock(num_channels_in, num_channels_output)
            rbs.append(rb)
        self.rbs = torch.nn.ModuleList(rbs)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, x: Tuple[torch.Tensor, List]) -> torch.Tensor:
        y, skips = x
        for rb in self.rbs:
            y = rb(y)
            skips.append(y)
        y = self.avg_pool(y)
        return y


class UpBlock(torch.nn.Module):

    def __init__(self, num_channels_input_signal: int, num_channels_input_skip: List[int], num_channels_output: int, block_depth: int):
        super(UpBlock, self).__init__()
        self.block_depth = block_depth
        rbs = []
        for i in range(block_depth):
            num_channels_in = num_channels_input_signal if i == 0 else num_channels_output
            num_channels_in += num_channels_input_skip[i]
            rb = ResidualBlock(num_channels_in, num_channels_output)
            rbs.append(rb)
        self.rbs = torch.nn.ModuleList(rbs)
        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: Tuple[torch.Tensor, List]) -> torch.Tensor:
        y, skips = x
        y = self.up(y)
        for rb in self.rbs:
            # Catenate along features
            y = torch.cat([y, skips.pop()], dim=1)
            y = rb(y)
        return y


class UNet(torch.nn.Module):

    def __init__(self, image_size: int, noise_embedding_size: int, num_channels_after_img_conv: int = 32):
        super(UNet, self).__init__()

        self.image_size = image_size
        self.noise_embedding_size = noise_embedding_size
        self.num_channels_after_img_conv = num_channels_after_img_conv
        self.conv_change_input = torch.nn.Conv2d(in_channels=3, out_channels=num_channels_after_img_conv, kernel_size=1)

        # Noise embeddings
        self.noise_embedding_up = torch.nn.Upsample(size=(image_size, image_size), mode='nearest')

        # Down blocks
        # Input size is the number of channels in the image + the number of channels in the noise embedding
        down_blocks = [
            DownBlock(num_channels_input=num_channels_after_img_conv+noise_embedding_size, num_channels_output=32, block_depth=2),
            DownBlock(num_channels_input=32, num_channels_output=64, block_depth=2),
            DownBlock(num_channels_input=64, num_channels_output=96, block_depth=2),
            ]
        self.down_blocks = torch.nn.ModuleList(down_blocks)

        # Residual blocks
        residual_blocks = [
            ResidualBlock(num_channels_input=96, num_channels_output=128),
            ResidualBlock(num_channels_input=128, num_channels_output=128),
            ]
        self.residual_blocks = torch.nn.ModuleList(residual_blocks)

        # Up blocks
        up_blocks = [
            UpBlock(num_channels_input_signal=128, num_channels_input_skip=[96,96], num_channels_output=96, block_depth=2),
            UpBlock(num_channels_input_signal=96, num_channels_input_skip=[64,64], num_channels_output=64, block_depth=2),
            UpBlock(num_channels_input_signal=64, num_channels_input_skip=[32,32], num_channels_output=32, block_depth=2),
            ]
        self.up_blocks = torch.nn.ModuleList(up_blocks)

        # Change shape to match image, init to zeros
        self.conv_change_output = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)
        # Initialize the convolutional layer's weights to zeros, underscore means in-place ie modify the tensor in-place
        torch.nn.init.zeros_(self.conv_change_output.weight)

    def forward(self, x: torch.Tensor, noise_variances: torch.Tensor) -> torch.Tensor:
        # Input shape: 
        # x: (batch_size, 3, image_size, image_size)
        # noise_variances: (batch_size, 1, 1, 1) 

        # Embedding of noise variances
        noise_embedding = sinusoidal_embedding(noise_variances, noise_embedding_size=self.noise_embedding_size) # (batch_size, noise_embedding_size, 1, 1)
        noise_embedding = self.noise_embedding_up(noise_embedding) # (batch_size, noise_embedding_size, image_size, image_size)
        assert noise_embedding.shape[1] == self.noise_embedding_size

        # Change image shape
        x = self.conv_change_input(x) # (batch_size, num_channels_after_img_conv, image_size, image_size)
        assert x.shape[1] == self.num_channels_after_img_conv

        # Concatenate noise embedding
        x = torch.cat([x, noise_embedding], dim=1) # (batch_size, num_channels_after_img_conv+noise_embedding_size, image_size, image_size)
        assert x.shape[1] == self.num_channels_after_img_conv + self.noise_embedding_size

        # Down blocks, with skips
        skips = []
        for db in self.down_blocks:
            x = db([x, skips])
        # Assume last block output channels = 96
        # Shape of x: (batch_size, 96, image_size // 8, image_size // 8)
            
        # Residual blocks
        for rb in self.residual_blocks:
            x = rb(x)
        # Assume last block output channels = 128
        # Shape of x: (batch_size, 128, image_size // 8, image_size // 8)

        # Up blocks, with skips
        for ub in self.up_blocks:
            x = ub([x, skips])
        # Assume last block output channels = 32
        # Shape of x: (batch_size, 32, image_size, image_size)
        
        # Convolve one more time to change shape back
        x = self.conv_change_output(x)

        return x