import torch
import math

def sinusoidal_embedding(x: torch.Tensor, noise_embedding_size: int = 32) -> torch.Tensor:
    frequencies = torch.exp(
        torch.linspace(
            start=math.log(1.0),
            end=math.log(1000.0),
            steps=noise_embedding_size // 2
            )
        )
    
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = torch.concat([
        torch.sin(angular_speeds * x), 
        torch.cos(angular_speeds * x)
        ], dim=3)
    return embeddings


def plot_sinusoidal_embeddings(noise_embedding_size: int = 32):
    import matplotlib.pyplot as plt
    import numpy as np

    embedding_list = []
    for y in np.arange(0, 1, 0.01):
        x = torch.tensor([[[[y]]]])
        embedding_list.append(sinusoidal_embedding(x, noise_embedding_size)[0][0][0])
    embedding_array = np.array(np.transpose(embedding_list))
    
    fig, ax = plt.subplots()
    ax.set_xticks(
        np.arange(0, 100, 10), labels=np.round(np.arange(0.0, 1.0, 0.1), 1)
    )
    ax.set_ylabel("embedding dimension", fontsize=8)
    ax.set_xlabel("noise variance", fontsize=8)
    plt.pcolor(embedding_array, cmap="coolwarm")
    plt.colorbar(orientation="horizontal", label="embedding value")
    ax.imshow(embedding_array, interpolation="nearest", origin="lower")
    plt.show()
