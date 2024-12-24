import random
import albumentations as A
from torchvision import transforms
import cv2
import numpy as np
import torch
from PIL import Image

## These are extra functions i had to create for trying to replicate sunny in rainy images ##

# This one modify bright and these kind of things
def add_sunlight(image, alpha=1.2, beta=50):
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return result

# Autoexplicative name
def remove_fog(image, clipLimit=2.0, tileGridSize=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result

# This one is tricky because there are a lot of reflections, not happy with the result, right now it inpaint the zones that are really bright
def remove_reflections(image, max_area=5000, th1=210, th2=255):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    s = cv2.medianBlur(s, 15)
    _, mask = cv2.threshold(v, th1, th2, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_mask = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= max_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    inpainted = cv2.inpaint(image, filtered_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return inpainted

def get_sunny_augmentation():
    sunny_augmentation = A.Compose([
            A.RandomRain(
                brightness_coefficient=random.choice([0.8, 0.85, 0.9]),
                drop_width=1,
                blur_value=5,
                p=1,
                rain_type=None
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 1),
                src_radius=1,
                num_flare_circles_lower=1,
                num_flare_circles_upper=4,
                p=0.6
            ),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.05), p=1),
            A.GaussianBlur(blur_limit=(3, 5), p=0.8),
        ])

    return sunny_augmentation

def get_rainy_augmentation():
    rainy_augmentation = lambda img: remove_fog(
        add_sunlight(remove_reflections(img, max_area=5000), alpha=1.2, beta=50),
        clipLimit=2.0, tileGridSize=8
    )

    return rainy_augmentation

# Main function for base transformations (+ some others if selected)
def get_transforms(input_size, general_transforms=False):
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Wrapper for Albumentations transforms so we can use it as transform(image)
    class AlbumentationsTransform:
        def __init__(self, albumentations_transform):
            self.albumentations_transform = albumentations_transform

        def __call__(self, image):
            image = np.array(image)
            augmented = self.albumentations_transform(image=image)
            return Image.fromarray(augmented['image'])

    # general transforms
    general_transform = None
    if general_transforms:
        general_transform = AlbumentationsTransform(A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
            A.HorizontalFlip(p=0.5)
        ]))

    # Define the final transformation
    final_transform = transforms.Compose([
        general_transform if general_transform else lambda x: x,
        base_transform
    ])

    # The transformation for mask requires to not normalize it, therefore it needs a custom way to get it into a tensor
    class ToClassTensor:
        def __call__(self, mask):
            return torch.tensor(np.array(mask), dtype=torch.long)

    segmentation_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        ToClassTensor()
    ])

    return final_transform, segmentation_transform
