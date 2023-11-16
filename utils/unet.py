import torch

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(torch.nn.Module):

    def __init__(self, num_channels_input: int, num_channels_output: int):
        super(ResidualBlock, self).__init__()
        self.input_width = num_channels_input
        self.width = num_channels_output

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
        if self.input_width != self.width:
            residual = self.conv_change_dim(x)
        else:
            residual = x

        x = self.batch_norm(x)
        x = self.conv_1(x)
        x = self.swish(x)
        x = self.conv_2(x)
        return x + residual