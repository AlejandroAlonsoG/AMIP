from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict

def load_config(file_path):
    """Loads a YAML file and converts it to an EasyDict."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return EasyDict(data)

def get_classes():
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

    return classes

def get_transforms():
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

    return image_transform, segmentation_transform

def load_loss(loss):
    if loss == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Requested loss not implemented!")
    
def load_optimizer(optimizer, parameters, lr):
    if optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=lr)
    else:
        raise NotImplementedError("Requested optimizer not implemented!")