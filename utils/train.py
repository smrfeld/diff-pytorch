from .load_dataset import make_dataloader_image_folder
from .diffusion_schedule import offset_cosine_diffusion_schedule
from .unet import UNet

from dataclasses import dataclass
from mashumaro import DataClassDictMixin
import torch
from typing import Tuple
from loguru import logger


@dataclass
class Conf(DataClassDictMixin):
    num_epochs: int
    output_dir: str
    image_folder: str
    image_size: int = 64
    dataset_repetitions: int = 5
    batch_size: int = 64
    noise_embedding_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "Adam"


class DiffusionModel:

    def __init__(self, conf: Conf):
        self.conf = conf

        # Make data loader
        self.dataloader = make_dataloader_image_folder(
            image_folder=conf.image_folder, 
            image_size=conf.image_size, 
            dataset_repetitions=conf.dataset_repetitions, 
            batch_size=conf.batch_size
            )

    def train(self):

        # Make the model
        model = UNet(
            image_size=self.conf.image_size, 
            noise_embedding_size=self.conf.noise_embedding_size
            )

        # Define the optimizer
        optimizer_class = getattr(torch.optim, self.conf.optimizer)
        optimizer = optimizer_class(model.parameters(), lr=self.conf.learning_rate)

        for epoch in range(self.conf.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.conf.num_epochs}")

            model.train()
            for input_images in self.dataloader:
                self._train_step(input_images, model, optimizer)


    def denoise(self, model: torch.nn.Module, noisy_images: torch.Tensor, noise_rates: torch.Tensor, signal_rates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_noises = model(noisy_images, noise_rates**2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images


    def _train_step(self, images: torch.Tensor, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        # Train mode
        model.train()

        # Get batch size
        batch_size = images.shape[0]

        # Sample diffusion times
        diffusion_times = torch.rand(batch_size, 1, 1, 1)
        noise_rates, signal_rates = offset_cosine_diffusion_schedule(diffusion_times)

        # Sample noise
        noises = torch.randn(batch_size, 3, self.conf.image_size, self.conf.image_size)

        # Corrupt images
        noisy_images = signal_rates * images + noise_rates * noises

        # Predict noise using UNet
        pred_noises, pred_images = self.denoise(model, noisy_images, noise_rates, signal_rates)

        # Compute loss = MSE
        loss = torch.mean((pred_noises - noises)**2)
        
        # Backpropagate
        loss.backward()

        # Update model parameters
        optimizer.step()


