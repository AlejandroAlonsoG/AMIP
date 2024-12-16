from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from dataset_utils import VancouverDataset, generate_splits
from unet import UNet
from trainer import BaselineTrainer

# Generate splits
root_dir = "/net/ens/am4ip/datasets/project-dataset"

classes = [
    ("Unlabeled", 0, (0, 0, 0)),
    ("Static", 4, (0, 0, 0)),
    ("Dynamic", 5, (111, 74, 0)),
    ("Ground", 6, (81, 0, 81)),
    ("Road", 7, (128, 64, 128)),
    ("Sidewalk", 8, (244, 35, 232)),
    ("Parking", 9, (250, 170, 160)),
    ("Rail track", 10, (230, 150, 140)),
    ("Building", 11, (70, 70, 70)),
    ("Wall", 12, (102, 102, 156)),
    ("Fence", 13, (190, 153, 153)),
    ("Guard rail", 14, (180, 165, 180)),
    ("Bridge", 15, (150, 100, 100)),
    ("Tunnel", 16, (150, 120, 90)),
    ("Pole", 17, (153, 153, 153)),
    ("Pole group", 18, (153, 153, 153)),
    ("Traffic light", 19, (250, 170, 30)),
    ("Traffic sign", 20, (220, 220, 0)),
    ("Vegetation", 21, (107, 142, 35)),
    ("Terrain", 22, (152, 251, 152)),
    ("Sky", 23, (70, 130, 180)),
    ("Person", 24, (220, 20, 60)),
    ("Rider", 25, (255, 0, 0)),
    ("Car", 26, (0, 0, 142)),
    ("Truck", 27, (0, 0, 70)),
    ("Bus", 28, (0, 60, 100)),
    ("Caravan", 29, (0, 0, 90)),
    ("Trailer", 30, (0, 0, 110)),
    ("Train", 31, (0, 80, 100)),
    ("Motorcycle", 32, (0, 0, 230)),
    ("Bicycle", 33, (119, 11, 32))
]

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class ToClassTensor:
    def __call__(self, mask):
        # Convert PIL image or NumPy array to a PyTorch tensor with integer values
        return torch.tensor(np.array(mask), dtype=torch.long)

segmentation_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),  # Use NEAREST to preserve class indices
    ToClassTensor()  # Convert to tensor without normalization
])


# Create datasets
train_dataset = VancouverDataset(
    root_dir=root_dir,
    split='train',
    groups=None,
    transform=image_transform,
    target_transform=segmentation_transform
)

test_dataset = VancouverDataset(
    root_dir=root_dir,
    split='test',
    groups=None,
    transform=image_transform,
    target_transform=segmentation_transform
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Verify splits
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Testing samples: {len(test_loader.dataset)}")

import matplotlib.pyplot as plt
import torchvision

def visualize_dataloader_samples(dataloader, num_samples=5):
    """
    Visualizes a few samples from a DataLoader.
    
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader to fetch samples from.
        classes (list): List of class names corresponding to the dataset.
        num_samples (int): Number of samples to visualize.
    """
    # Get a batch of data
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Select the first `num_samples` images
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Prepare the figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    for i in range(num_samples):
        img = images[i]
        label = labels[i]
        
        # Unnormalize the image
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format for display
        label = label.permute(1, 2, 0).numpy()  # Convert to HWC format for display
        
        # Display the image
        axes[i].imshow(label)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_dataloader_samples(train_loader, num_samples=5)
