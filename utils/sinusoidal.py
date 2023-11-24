import torch
import math

def sinusoidal_embedding(noise_variances: torch.Tensor, noise_embedding_size: int = 32) -> torch.Tensor:
    """Creates a sinusoidal embedding of the input tensor.

    Args:
        noise_variances (torch.Tensor): Noise variances, expected shape (batch_size, 1, 1, 1).
        noise_embedding_size (int, optional): Noise embedding dimension. Defaults to 32.

    Returns:
        torch.Tensor: Embeddings of shape: (batch_size, noise_embedding_size, 1, 1)
    """    

    # Input shape: noise_variances = (batch_size, 1, 1, 1)
    assert len(noise_variances.shape) == 4, f"Expected shape (batch_size, 1, 1, 1), got {noise_variances.shape}"
    assert noise_variances.shape[1] == 1, f"Expected shape (batch_size, 1, 1, 1), got {noise_variances.shape}"
    assert noise_variances.shape[2] == 1, f"Expected shape (batch_size, 1, 1, 1), got {noise_variances.shape}"
    assert noise_variances.shape[3] == 1, f"Expected shape (batch_size, 1, 1, 1), got {noise_variances.shape}"

    frequencies = torch.exp(
        torch.linspace(
            start=math.log(1.0),
            end=math.log(1000.0),
            steps=noise_embedding_size // 2
            )
        )
    frequencies = frequencies.to(noise_variances.device) # shape = [noise_embedding_size // 2]
    
    angular_speeds = 2.0 * math.pi * frequencies # shape = [noise_embedding_size // 2]
    
    # Reshape angular speeds to [1, noise_embedding_size // 2, 1, 1]
    angular_speeds_reshaped = angular_speeds.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # shape = [1, noise_embedding_size // 2, 1, 1]

    sin = torch.sin(angular_speeds_reshaped * noise_variances) # shape = [batch_size, noise_embedding_size // 2, 1, 1]
    cos = torch.cos(angular_speeds_reshaped * noise_variances) # shape = [batch_size, noise_embedding_size // 2, 1, 1]
    embeddings = torch.concat([sin,cos], dim=1) # shape = [batch_size, noise_embedding_size, 1, 1]
    return embeddings


def plot_sinusoidal_embeddings(noise_embedding_size: int = 32):
    """Plots the sinusoidal embeddings for different noise variances.

    Args:
        noise_embedding_size (int, optional): Noise embedding size. Defaults to 32.
    """   
    import plotly.graph_objs as go
    import numpy as np

    embedding_list = []
    for y in np.arange(0, 1, 0.01):
        x = torch.tensor([[[[y]]]])
        embedding_list.append(sinusoidal_embedding(x, noise_embedding_size)[0][0][0])
    embedding_array = np.array(np.transpose(embedding_list))

    # Create a heatmap using Plotly
    heatmap = go.Heatmap(
        z=embedding_array,
        x=np.round(np.arange(0.0, 1.0, 0.1), 1),
        y=np.arange(0, 100, 10),
        colorscale="coolwarm",
    )

    # Create the layout
    layout = go.Layout(
        xaxis=dict(
            title="noise variance",
            tickfont=dict(size=8),
            tickvals=np.arange(0, 1, 0.1),
            ticktext=np.round(np.arange(0.0, 1.0, 0.1), 1),
        ),
        yaxis=dict(title="embedding dimension", tickfont=dict(size=8)),
        coloraxis=dict(colorbar=dict(orientation="h", title="embedding value")),
    )

    # Create the figure
    fig = go.Figure(data=[heatmap], layout=layout)

    # Show the plot
    fig.show()
