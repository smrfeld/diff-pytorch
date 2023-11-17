from .load_dataset import make_dataloader_image_folder, image_postprocess_fn
from .diffusion_schedule import offset_cosine_diffusion_schedule
from .unet import UNet

from dataclasses import dataclass
from mashumaro import DataClassDictMixin
import torch
from typing import Tuple, Optional, List
from loguru import logger
from PIL import Image
import numpy as np


class DiffusionModel:


    @dataclass
    class Conf(DataClassDictMixin):
        num_epochs: int
        output_dir: str
        image_folder_train: str
        image_folder_test: Optional[str] = None
        image_size: int = 64
        dataset_repetitions: int = 5
        batch_size: int = 64
        noise_embedding_size: int = 32
        num_epochs: int = 10
        learning_rate: float = 0.001
        optimizer: str = "Adam"
        random_seed: int = 42
        validation_split: float = 0.2

        def check_valid(self):
            assert 0 <= self.validation_split <= 1, "validation_split must be between 0 and 1"


    def __init__(self, conf: Conf):
        conf.check_valid()
        self.conf = conf
        self.model = None


    def post_process_images(self, images: torch.Tensor) -> torch.Tensor:
        transform = image_postprocess_fn()
        images = transform(images)
        return torch.clip(images, 0.0, 1.0)


    def generate(self, num_images: int = 1, diffusion_steps: int = 20, initial_noise: Optional[torch.Tensor] = None) -> List[Image.Image]:
        if initial_noise is None:
            initial_noise = torch.randn(
                size=(num_images, 3, self.conf.image_size, self.conf.image_size)
            )
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)

        # Undo the image preprocessing
        generated_images = self.post_process_images(generated_images)

        # Convert to PIL images
        generated_images = [
            Image.fromarray((255.0 * image.numpy()).astype(np.uint8))
            for image in generated_images
            ]

        return generated_images


    def reverse_diffusion(self, initial_noise: torch.Tensor, diffusion_steps: int) -> torch.Tensor:
        model = self._make_model_if_needed()

        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        pred_images = None
        for step in range(diffusion_steps):

            # Compute diffusion times
            diffusion_times = torch.ones((num_images, 1, 1, 1)) - step * step_size

            # Compute noise and signal rates
            noise_rates, signal_rates = offset_cosine_diffusion_schedule(diffusion_times)

            # Predict the noise
            pred_noises, pred_images = self.denoise(model, current_images, noise_rates, signal_rates)

            # Predict the next point (t-1) from this one (t): 
            # (1) subtract the current predicted noise to get the predicted image at time (0)
            # (2) at that point, compute what the noise would be at the next step (t-1) from the time (0) one
            # (3) add that noise to the predicted image at time (0) to get the predicted image at time (t-1)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = offset_cosine_diffusion_schedule(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        
        assert pred_images is not None, "No diffusion steps were performed"
        return pred_images


    def _make_model_if_needed(self) -> torch.nn.Module:
        if self.model is None:
            self.model = UNet(
                image_size=self.conf.image_size, 
                noise_embedding_size=self.conf.noise_embedding_size
                )
        return self.model


    def train(self):

        # Make data loader for train and val splits
        train_loader, val_loader = make_dataloader_image_folder(
            image_folder=self.conf.image_folder_train, 
            image_size=self.conf.image_size, 
            dataset_repetitions=self.conf.dataset_repetitions, 
            batch_size=self.conf.batch_size,
            validation_split=self.conf.validation_split,
            random_seed=self.conf.random_seed
            )

        # Make the model
        model = self._make_model_if_needed()

        # Define the optimizer
        optimizer_class = getattr(torch.optim, self.conf.optimizer)
        optimizer = optimizer_class(model.parameters(), lr=self.conf.learning_rate)

        for epoch in range(self.conf.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.conf.num_epochs}")

            # Take a training step
            train_loss = 0.0
            for input_image_batch in train_loader:                
                train_loss += self._train_step(input_image_batch, optimizer, model).item()
            
            # Compute loss from validation set
            val_loss = 0.0
            for input_image_batch in val_loader:
                model.eval()
                val_loss += self._compute_loss(input_image_batch, model).item()

            # Average
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")


    def denoise(self, model: torch.nn.Module, noisy_images: torch.Tensor, noise_rates: torch.Tensor, signal_rates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:        
        noise_variances = noise_rates**2
        pred_noises = model(noisy_images, noise_variances)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images


    def _compute_loss(self, images: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:

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
        return loss


    def _train_step(self, images: torch.Tensor, optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> torch.Tensor:
        # Train mode
        model.train()

        loss = self._compute_loss(images, model)

        # Backpropagate
        loss.backward()

        # Update model parameters
        optimizer.step()

        return loss


