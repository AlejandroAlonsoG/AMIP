import os
from torch.utils.data import Dataset
from PIL import Image
import random

import os
import random
from torch.utils.data import Dataset
from PIL import Image

class VancouverDataset(Dataset):
    def __init__(self, root_dir, split, groups=None, transform=None, target_transform=None, split_ratio=0.8):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            split (str): 'train' or 'test'.
            groups (dict, optional): Mapping of image IDs to their split ('train' or 'test').
            transform (callable, optional): A function/transform to apply to the images.
            target_transform (callable, optional): A function/transform to apply to the segmentation maps.
            split_ratio (float, optional): Ratio of training data to total data if creating own splits.
        """
        self.root_dir = root_dir
        self.split = split
        self.groups = groups
        self.transform = transform
        self.target_transform = target_transform
        self.split_ratio = split_ratio
        
        # Combine all data
        self.data = []
        for mode in ['rainy', 'sunny']:
            image_dir = os.path.join(root_dir, f"{mode}_images")
            sseg_dir = os.path.join(root_dir, f"{mode}_sseg")
            for img_file in os.listdir(image_dir):
                img_id = os.path.splitext(img_file)[0]
                self.data.append({
                    'image_path': os.path.join(image_dir, img_file),
                    'sseg_path': os.path.join(sseg_dir, os.path.splitext(img_file)[0] + ".png"),
                    'id': img_id
                })
        
        # Create splits if no groups provided
        if self.groups is None:
            random.seed(42)
            random.shuffle(self.data) # Always same split
            split_idx = int(len(self.data) * self.split_ratio)
            if self.split == 'train':
                self.data = self.data[:split_idx]
            elif self.split == 'test':
                self.data = self.data[split_idx:]
            else:
                raise ValueError("split must be 'train' or 'test'")
        else:
            # Filter data based on the provided groups
            self.data = [item for item in self.data if self.groups.get(item['id']) == split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert("RGB")
        segmentation = Image.open(item['sseg_path'])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            segmentation = self.target_transform(segmentation)

        return image, segmentation


def generate_splits(root_dir, sunny_groups, rainny_groups, test_ratio=0.2, seed=42):
    """
    Args:
        root_dir (str): Path to the dataset directory.
        sunny_groups (dict): Mapping of sunny image IDs to groups (e.g., "group1", "group2").
        test_ratio (float): Proportion of the dataset to use for testing.
        seed (int): Random seed for reproducibility.
    
    Returns:
        dict: Mapping of all image IDs to their split ('train' or 'test').
    """
    random.seed(seed)

    # Gather all image IDs
    groups = {}
    ignored = {}
    for mode in ['rainy', 'sunny']:
        image_dir = os.path.join(root_dir, f"{mode}_images")
        ignored[mode] = []
        for img_file in os.listdir(image_dir):
            img_id = os.path.splitext(img_file)[0]
            if mode == 'sunny' and img_id in sunny_groups:
                groups.setdefault(sunny_groups[img_id], []).append(img_id)
            elif mode == 'rainy' and img_id in rainny_groups:
                groups.setdefault(rainny_groups[img_id], []).append(img_id)
            else:
                ignored[mode].append(img_id)
    
    # Split groups into train and test
    splits = {}
    for group, img_ids in groups.items():
        is_test = random.random() < test_ratio
        for img_id in img_ids:
            splits[img_id] = 'test' if is_test else 'train'
    
    for group, img_ids in ignored.items():
        print(f"Ignored the following {group} images:")
        print(img_ids)

    return splits
