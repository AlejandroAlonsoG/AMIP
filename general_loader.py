import torch
import torch.nn as nn
import torch.nn.functional as F
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

def load_loss(loss):
    if loss == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif loss == 'FocalLoss':
        return FocalLoss()
    else:
        raise NotImplementedError("Requested loss not implemented!")
    
def load_optimizer(optimizer, parameters, lr):
    if optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=lr)
    else:
        raise NotImplementedError("Requested optimizer not implemented!")

# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)  # Number of classes
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Compute probabilities
        probs = torch.exp(log_probs)
        
        # Compute focal weights
        focal_weights = self.alpha * (1 - probs) ** self.gamma
        
        # Compute focal loss
        loss = -focal_weights * targets_one_hot * log_probs
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.sum(dim=1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss