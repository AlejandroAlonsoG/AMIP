
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from dataset_utils import VancouverDataset, generate_splits
from unet import UNet
from trainer import BaselineTrainer

# Parameters
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

batch_size = 32
num_epochs = 25
learning_rate = 1e-3

# Transformation for images
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
# Transformation for mask requires to not normalize it, therefore it needs a custom way to get it into a tensor
class ToClassTensor:
    def __call__(self, mask):
        return torch.tensor(np.array(mask), dtype=torch.long)

segmentation_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
    ToClassTensor()
])

# Load datasets
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Training samples: {len(train_loader.dataset)}")
print(f"Testing samples: {len(test_loader.dataset)}")

# Define what we use for the training

class_mapping = [cls[1] for cls in classes]

model = UNet(num_classes=len(classes), class_mapping=class_mapping, input_channels=3)

criterion = nn.CrossEntropyLoss() # TODO try focal loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train

trainer = BaselineTrainer(
    model=model,
    loss=criterion,
    optimizer=optimizer,
    use_cuda=torch.cuda.is_available()
)

trainer.fit(train_data_loader=train_loader, epoch=num_epochs)

# Evaluate

trainer.evaluate(test_loader, num_classes=len(classes))