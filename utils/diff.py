from .load_dataset import make_dataloader_image_folder, image_postprocess_fn
from .diffusion_schedule import offset_cosine_diffusion_schedule
from .unet import UNet

from dataclasses import dataclass
from mashumaro import DataClassDictMixin
import torch
from typing import Tuple, Optional, List, Dict
from loguru import logger
from PIL import Image
import numpy as np
import os
from enum import Enum
from tqdm import tqdm
import json
import time
import shutil
import yaml


class DiffusionModel:
    """Diffusion model
    """    

    @dataclass
    class Conf(DataClassDictMixin):
        """Configuration for the diffusion model
        """        

        class Initialize(Enum):
            FROM_LATEST_CHECKPOINT = "from-latest-checkpoint"
            "Load the model from the latest checkpoint"

            FROM_BEST_CHECKPOINT = "from-best-checkpoint"
            "Load the model from the best checkpoint"

            FROM_SCRATCH = "from-scratch"
            "Initialize the model from scratch"

        output_dir: str
        "Directory where to save checkpoints and generated images"

        image_folder_train: str
        "Directory containing training images"

        device: str = "mps"
        "Device to use for training"

        image_size: int = 64
        "Size of the images in pixels"

        dataset_repetitions: int = 5
        "Number of times to repeat the dataset"

        batch_size: int = 64
        "Batch size"

        noise_embedding_size: int = 32
        "Size of the noise embedding"

        num_epochs: int = 10
        "Number of epochs to train for"

        learning_rate: float = 0.001
        "Learning rate"

        optimizer: str = "Adam"
        "Optimizer to use"

        random_seed: int = 42
        "Random seed for reproducibility"

        validation_split: float = 0.2
        "Fraction of the dataset to use for validation"

        initialize: Initialize = Initialize.FROM_SCRATCH
        "How to initialize the model"

        generate_no_images: int = 20
        "Number of images to generate"

        generate_diffusion_steps: int = 20
        "Number of diffusion steps to use when generating"

        class Loss(Enum):
            MSE = "mse"
            "Mean squared error"

            MAE = "mae"
            "Mean absolute error"

        loss: Loss = Loss.MSE
        "Loss function to use"

        def update_paths(self, mnt_dir: Optional[str]):
            if mnt_dir is not None:
                self.image_folder_train = os.path.join(mnt_dir, self.image_folder_train)

        @property
        def checkpoint_init(self):
            os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, "init.pth")

        @property
        def checkpoint_best(self):
            os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, "best.pth")

        @property
        def checkpoint_latest(self):
            os.makedirs(self.output_dir, exist_ok=True)
            return os.path.join(self.output_dir, "latest.pth")

        def check_valid(self):
            assert 0 <= self.validation_split <= 1, "validation_split must be between 0 and 1"

        @property
        def img_output_dir(self):
            import datetime
            now = datetime.datetime.now()
            return os.path.join(self.output_dir, f"images_{now.strftime('%Y%m%d_%H%M%S')}")

        @property
        def training_metadata_json(self):
            return os.path.join(self.output_dir, "training_metadata.json")


    @dataclass
    class TrainingMetadata(DataClassDictMixin):
        """Training metadata
        """        

        @dataclass
        class Metadata(DataClassDictMixin):
            """Metadata for a single epoch
            """            

            epoch: int
            "Epoch number"

            val_loss: float
            "Validation loss"

            train_loss: float
            "Training loss"

        epoch_to_metadata: Dict[int, Metadata]
        "Mapping from epoch number to training metadata"


    def load_training_metadata_or_new(self, epoch_start: Optional[int] = None) -> TrainingMetadata:
        fname = os.path.join(self.conf.output_dir, "training_metadata.json")
        if os.path.exists(fname):
            with open(fname, "r") as f:
                metadata = DiffusionModel.TrainingMetadata.from_dict(json.load(f))
            logger.info(f"Loaded training metadata from {fname}")

            # Clear epochs that are after the start epoch
            if epoch_start is not None:
                metadata.epoch_to_metadata = {
                    epoch: m for epoch, m in metadata.epoch_to_metadata.items() if epoch < epoch_start
                    }
            return metadata
        else:
            return DiffusionModel.TrainingMetadata(epoch_to_metadata={})


    def __init__(self, conf: Conf):
        conf.check_valid()
        self.conf = conf
        self.model = UNet(
            image_size=self.conf.image_size, 
            noise_embedding_size=self.conf.noise_embedding_size
            )
        # Send to device
        self.model = self.model.to(self.conf.device)

        if self.conf.initialize == self.conf.Initialize.FROM_LATEST_CHECKPOINT:
            self.load_checkpoint(self.conf.checkpoint_latest)
        elif self.conf.initialize == self.conf.Initialize.FROM_BEST_CHECKPOINT:
            self.load_checkpoint(self.conf.checkpoint_best)
        elif self.conf.initialize == self.conf.Initialize.FROM_SCRATCH:
            pass
        else:
            raise ValueError(f"Unknown train_from value {self.conf.initialize}")


    def post_process_images(self, images: torch.Tensor) -> torch.Tensor:
        transform = image_postprocess_fn()
        images = transform(images)
        return torch.clip(images, 0.0, 1.0)


    def generate(self, 
        num_images: Optional[int] = None, 
        diffusion_steps: Optional[int] = None, 
        initial_noise: Optional[torch.Tensor] = None
        ) -> List[Image.Image]:
        self.model.eval()

        if initial_noise is None:
            num_images = num_images or self.conf.generate_no_images
            initial_noise = torch.randn(
                size=(num_images, 3, self.conf.image_size, self.conf.image_size)
            )
        diffusion_steps = diffusion_steps or self.conf.generate_diffusion_steps
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)

        # Undo the image preprocessing
        generated_images = self.post_process_images(generated_images)

        # Convert to PIL images
        generated_images_pil = []
        num_images = num_images or self.conf.generate_no_images
        for i in range(num_images):
            img_np_arr = generated_images[i].permute(1,2,0).detach().cpu().numpy()
            img_np_arr *= 255
            img_np_arr = np.uint8(img_np_arr)
            img_pil = Image.fromarray(img_np_arr)
            generated_images_pil.append(img_pil)

        return generated_images_pil


    def reverse_diffusion(self, initial_noise: torch.Tensor, diffusion_steps: int) -> torch.Tensor:
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
            pred_noises, pred_images = self.denoise(self.model, current_images, noise_rates, signal_rates)

            # Predict the next point (t-1) from this one (t): 
            # (1) subtract the current predicted noise to get the predicted image at time (0)
            # (2) at that point, compute what the noise would be at the next step (t-1) from the time (0) one
            # (3) add that noise to the predicted image at time (0) to get the predicted image at time (t-1)

            pred_images = pred_images.to(self.conf.device)
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = offset_cosine_diffusion_schedule(next_diffusion_times)
            next_signal_rates = next_signal_rates.to(self.conf.device)
            next_noise_rates = next_noise_rates.to(self.conf.device)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        
        assert pred_images is not None, "No diffusion steps were performed"
        return pred_images


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

        # Define the optimizer
        optimizer_class = getattr(torch.optim, self.conf.optimizer)
        optimizer = optimizer_class(self.model.parameters(), lr=self.conf.learning_rate)

        # Load the model from the latest checkpoint
        if self.conf.initialize == self.conf.Initialize.FROM_LATEST_CHECKPOINT:
            epoch_start, val_loss_best = self.load_checkpoint(self.conf.checkpoint_latest, optimizer)
        elif self.conf.initialize == self.conf.Initialize.FROM_BEST_CHECKPOINT:
            epoch_start, val_loss_best = self.load_checkpoint(self.conf.checkpoint_best, optimizer)
        elif self.conf.initialize == self.conf.Initialize.FROM_SCRATCH:
            if os.path.exists(self.conf.output_dir):
                logger.warning("Initializing model from scratch. Erasing output directory. You have 8 seconds.")
                for _ in tqdm(range(8)):
                    time.sleep(1)
                shutil.rmtree(self.conf.output_dir)
            os.makedirs(self.conf.output_dir, exist_ok=True)

            epoch_start = 0
            val_loss_best = float("inf")

            # Write the initial model to disk
            self._save_checkpoint(epoch_start, optimizer, val_loss_best, self.conf.checkpoint_init)
        else:
            raise ValueError(f"Unknown train_from value {self.conf.initialize}")

        if epoch_start >= self.conf.num_epochs - 1:
            logger.info("Training already completed")
            return

        # Load training metadata
        metadata = self.load_training_metadata_or_new(epoch_start)

        # Copy config file to output directory
        fname = os.path.join(self.conf.output_dir, "conf.yml")
        with open(fname, "w") as f:
            yaml.dump(self.conf.to_dict(), f, indent=3)
            logger.info(f"Wrote config to {f.name}")

        for epoch in range(epoch_start, self.conf.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.conf.num_epochs}")
            t0 = time.time()

            # Take a training step
            train_loss = 0.0
            for input_image_batch, _ in tqdm(train_loader, desc="Training batch"):
                train_loss += self._train_step(input_image_batch, optimizer).item()
            
            # Compute loss from validation set
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for input_image_batch, _ in tqdm(val_loader, desc="Val batch"):
                    val_loss += self._compute_loss(input_image_batch).item()

            # Average
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            dur = time.time() - t0
            logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f} duration: {dur:.2f}s")

            # Store metadata
            metadata.epoch_to_metadata[epoch] = DiffusionModel.TrainingMetadata.Metadata(
                epoch=epoch,
                val_loss=val_loss,
                train_loss=train_loss
                )

            # Check to save
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                self._save_checkpoint(epoch, optimizer, val_loss_best, self.conf.checkpoint_best)
            self._save_checkpoint(epoch, optimizer, val_loss, self.conf.checkpoint_latest)
            self._write_training_metadata(metadata)


    def denoise(self, model: torch.nn.Module, noisy_images: torch.Tensor, noise_rates: torch.Tensor, signal_rates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:       
        # Inputs
        # noisy_images: (batch_size, 3, image_size, image_size)
        # noise_rates: (batch_size, 1, 1, 1)
        # signal_rates: (batch_size, 1, 1, 1) 

        # Variances
        noise_variances = noise_rates**2 # (batch_size, 1, 1, 1)

        # Send to device
        noisy_images = noisy_images.to(self.conf.device)
        noise_variances = noise_variances.to(self.conf.device)
        noise_rates = noise_rates.to(self.conf.device)
        signal_rates = signal_rates.to(self.conf.device)

        # Predict noise
        pred_noises = model(noisy_images, noise_variances) # (batch_size, 3, image_size, image_size)

        # Predict image
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates # (batch_size, 3, image_size, image_size)
        return pred_noises, pred_images


    def _save_checkpoint(self, epoch: int, optimizer: torch.optim.Optimizer, loss: float, fname: str):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, fname)


    def load_checkpoint(self, fname: str, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int,float]:
        logger.debug(f"Loading checkpoint from {fname}")
        checkpoint = torch.load(fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"Loaded optimizer state from epoch {epoch} with loss {loss}")
        return epoch, loss


    def _write_training_metadata(self, metadata: TrainingMetadata):
        with open(self.conf.training_metadata_json, "w") as f:
            json.dump(metadata.to_dict(), f, indent=3)
            logger.info(f"Wrote training metadata to {f.name}")


    def _compute_loss(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.conf.device)

        # Get batch size
        batch_size = images.shape[0]

        # Sample diffusion times
        diffusion_times = torch.rand(batch_size, 1, 1, 1) # (batch_size, 1, 1, 1)
        noise_rates, signal_rates = offset_cosine_diffusion_schedule(diffusion_times) # (batch_size, 1, 1, 1)
        noise_rates = noise_rates.to(self.conf.device)
        signal_rates = signal_rates.to(self.conf.device)

        # Sample noise
        noises = torch.randn(batch_size, 3, self.conf.image_size, self.conf.image_size) # (batch_size, 3, image_size, image_size)
        noises = noises.to(self.conf.device)

        # Corrupt images
        noisy_images = signal_rates * images + noise_rates * noises # (batch_size, 3, image_size, image_size)

        # Predict noise using UNet
        pred_noises, pred_images = self.denoise(self.model, noisy_images, noise_rates, signal_rates) # (batch_size, 3, image_size, image_size)

        # Compute loss
        if self.conf.loss == DiffusionModel.Conf.Loss.MSE:
            # MSE
            loss = torch.mean((pred_noises - noises)**2)
        elif self.conf.loss == DiffusionModel.Conf.Loss.MAE:
            # MAE error
            loss = torch.mean(torch.abs(pred_noises - noises))
        else:
            raise ValueError(f"Unknown loss {self.conf.loss}")
        return loss


    def _train_step(self, images: torch.Tensor, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        # Reset gradients
        optimizer.zero_grad()

        # Train mode
        self.model.train()

        # Compute loss
        loss = self._compute_loss(images)

        # Backpropagate
        loss.backward()

        # Update model parameters
        optimizer.step()

        return loss


