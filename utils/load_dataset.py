from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms

def make_dataloader_image_folder(image_folder: str, image_size: int = 64, dataset_repetitions: int = 5, batch_size: int = 64) -> DataLoader:

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize images to a fixed size
        transforms.RandomHorizontalFlip(),  # Apply random horizontal flip for data augmentation
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
        ])
    dataset = ImageFolder(root=image_folder, transform=transform)
    
    # Create a list of 'dataset' repeated 'dataset_repetitions' times
    repeated_datasets = [dataset] * dataset_repetitions
    concatenated_dataset = ConcatDataset(repeated_datasets)

    return DataLoader(concatenated_dataset, batch_size=batch_size, shuffle=True)