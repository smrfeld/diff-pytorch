import sys
sys.path.append("..")

import torch
from utils import ResidualBlock

def test_residual_block():
    num_channels_input = 3
    num_channels_output = 6
    batch_size = 16
    height = 8
    width = 8

    rb = ResidualBlock(
        num_channels_input=num_channels_input,
        num_channels_output=num_channels_output
        )

    # Input = (batch_size, num_channels, height, width)
    input_tensor = torch.randn(batch_size, num_channels_input, height, width)
    output = rb(input_tensor)
    
    # Check if the output tensor has the correct shape
    assert output.shape == torch.Size([batch_size, num_channels_output, height, width])