from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms
import torch
from typing import Tuple


def image_preprocess_fn(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize images to a fixed size
        transforms.RandomHorizontalFlip(),  # Apply random horizontal flip for data augmentation
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
        ])


def image_postprocess_fn() -> transforms.Compose:
    return transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            )
        ])


def make_dataloader_image_folder(
    image_folder: str, 
    image_size: int = 64, 
    dataset_repetitions: int = 5, 
    batch_size: int = 64,
    validation_split: float = 0.2, 
    random_seed: int = 42
    ) -> Tuple[DataLoader,DataLoader]:
    transform = image_preprocess_fn(image_size)
    dataset = ImageFolder(root=image_folder, transform=transform)
    
    # Create a list of 'dataset' repeated 'dataset_repetitions' times
    repeated_datasets = [dataset] * dataset_repetitions
    concatenated_dataset = ConcatDataset(repeated_datasets)

    # Calculate the number of samples for validation
    num_samples = len(concatenated_dataset)
    num_val_samples = int(validation_split * num_samples)
    num_train_samples = num_samples - num_val_samples

    # Use a fixed random seed for reproducibility
    torch.manual_seed(random_seed)

    # Split the dataset into training and validation sets
    train_indices = list(range(num_train_samples))
    val_indices = list(range(num_train_samples, num_samples))

    train_dataset = Subset(concatenated_dataset, train_indices)
    val_dataset = Subset(concatenated_dataset, val_indices)

    # Create DataLoader instances for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
