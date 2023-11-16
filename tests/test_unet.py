import sys
sys.path.append("..")

import torch
from utils import ResidualBlock, DownBlock, UpBlock, UNet

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

def test_down_block():
    num_channels_input = 3
    num_channels_output = 6
    block_depth = 4
    batch_size = 16
    height = 8
    width = 8

    db = DownBlock(
        num_channels_input=num_channels_input,
        num_channels_output=num_channels_output,
        block_depth=block_depth
        )
    
    # Input = (batch_size, num_channels, height, width)
    input_tensor = torch.randn(batch_size, num_channels_input, height, width)
    skips = []
    output = db((input_tensor, skips))

    assert output.shape == torch.Size([batch_size, num_channels_output, height // 2, width // 2])

def test_up_block():
    num_channels_input = 3
    num_channels_output = 6
    block_depth = 4
    batch_size = 16
    height = 8
    width = 8

    db = DownBlock(
        num_channels_input=num_channels_input,
        num_channels_output=num_channels_output,
        block_depth=block_depth
        )
    
    # Input = (batch_size, num_channels, height, width)
    input_tensor = torch.randn(batch_size, num_channels_input, height, width)
    skips = []
    output_tensor = db((input_tensor, skips))

    num_channels_input = output_tensor.shape[1]
    ub = UpBlock(
        num_channels_input_signal=num_channels_input,
        num_channels_input_skip=block_depth*[num_channels_output],
        num_channels_output=num_channels_output,
        block_depth=block_depth
        )
    
    # Input = (batch_size, num_channels, height // 2, width // 2)
    output = ub((output_tensor, skips))

    assert output.shape == torch.Size([batch_size, num_channels_output, height, width])


def test_unet():
    image_size = 64
    batch_size = 8

    unet = UNet(
        image_size=image_size,
        noise_embedding_size=32
        )

    # Print summary of model
    # from torchsummary import summary
    # summary(unet, input_size=[(batch_size, 3, image_size, image_size), (batch_size, 1,1,1)])

    '''
    diffusion_times = tf.random.uniform(
        shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0
    )
    noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
    '''

    noise_rates = torch.randn(batch_size, 1, 1, 1)
    noise_variances = noise_rates**2

    # Fake image
    input_tensor = torch.randn(batch_size, 3, image_size, image_size)

    # Output
    output_tensor = unet(input_tensor, noise_variances)

    # Shape
    assert output_tensor.shape == torch.Size([batch_size,3,image_size,image_size])