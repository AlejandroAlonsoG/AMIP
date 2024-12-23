import cv2
import numpy as np
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, GaussianBlur, RandomRain, RandomCrop, RandomSunFlare, RandomFog
from torchvision import transforms
from PIL import Image
import torch
import random

# Functions for weather-specific augmentations
def add_sunlight(image, alpha=1.2, beta=50):
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return result

def remove_fog(image, clipLimit=2.0, tileGridSize=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result

import cv2
import numpy as np

def remove_reflections(image, max_area=5000):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Smooth the Saturation channel
    s = cv2.medianBlur(s, 15)
    
    # Threshold the Value channel
    _, mask = cv2.threshold(v, 210, 255, cv2.THRESH_BINARY)  # Adjust threshold (240) as needed
    
    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a new mask with filtered contours
    filtered_mask = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= max_area:  # Keep only small areas
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Inpaint the original image using the filtered mask
    inpainted = cv2.inpaint(image, filtered_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    
    return inpainted


# Transformation functions
def get_transforms(general_augmentation=False, weather_augmentation=False):
    if not general_augmentation and not weather_augmentation:
        image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image_rain_transform = image_transform
        image_sun_transform = image_transform

    elif not weather_augmentation:
        image_transform = Compose([
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
            HorizontalFlip(p=0.5),
            RandomCrop(width=256, height=256, p=1.0)
        ])
        image_rain_transform = image_transform
        image_sun_transform = image_transform

    elif not general_augmentation:
        image_rain_transform = Compose([
            RandomRain(p=0.7),
            RandomBrightnessContrast(p=0.7),
            GaussianBlur(p=0.5)
        ])
        
        # Sunny transformation pipeline using manual functions
        def sunny_pipeline(image):
            image = apply_sunny_transformations(image)
            image = cv2.resize(image, (256, 256))
            image = torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
            return image

        image_sun_transform = sunny_pipeline

    else:
        # Merge general and weather augmentations
        image_rain_transform = Compose([
            RandomBrightnessContrast(p=0.7),
            HorizontalFlip(p=0.5),
            RandomRain(p=0.7),
            GaussianBlur(p=0.5),
            RandomCrop(width=256, height=256, p=1.0)
        ])
        
        def sunny_pipeline(image):
            image = cv2.resize(image, (256, 256))
            image = torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
            return image

        image_sun_transform = sunny_pipeline

    # Transformation for segmentation masks
    class ToClassTensor:
        def __call__(self, mask):
            return torch.tensor(np.array(mask), dtype=torch.long)

    segmentation_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        ToClassTensor()
    ])

    return (image_rain_transform, image_sun_transform), segmentation_transform

def manual_rain():
    transform = Compose([
        RandomRain(brightness_coefficient=random.choice([0.8, 0.85, 0.9]), drop_width=1, blur_value=5, p=1, rain_type=None),
        RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=1, num_flare_circles_lower=1, num_flare_circles_upper=4, p=0.6),
        RandomBrightnessContrast(brightness_limit=(-0.1,-0.05),p=1),
        GaussianBlur(p=0.8)
    ])
if __name__ == "__main__":
    import os

    # Example file paths
    sunny_image_path = "/net/ens/am4ip/datasets/project-dataset/sunny_images/00067086.png"
    rainy_image_path = "/net/ens/am4ip/datasets/project-dataset/rainy_images/00055371.png"

    # Load example images
    rain_image = cv2.imread(rainy_image_path)
    rain_image = cv2.cvtColor(rain_image, cv2.COLOR_BGR2RGB)

    # Apply transformations and save images
    tmp2 = 0
    for tmp1 in [0.8]:
        for tmp2 in [30, 40, 50]:
            output_path = f"output_manual_{tmp1}_{tmp2}.jpg"
            transformed = remove_reflections(rain_image, max_area=5000)
            transformed =  add_sunlight(transformed, alpha=0.8, beta=50)
            transformed = remove_fog(transformed, clipLimit=2.0, tileGridSize=16)
            cv2.imwrite(output_path, cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))


    """
    Sun transform:
    Compose([
        RandomRain(brightness_coefficient=random.choice([0.8, 0.85, 0.9]), drop_width=1, blur_value=5, p=1, rain_type=None),
        RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=1, num_flare_circles_lower=1, num_flare_circles_upper=4, p=0.6),
        RandomBrightnessContrast(brightness_limit=(-0.1,-0.05),p=1),
        GaussianBlur(p=0.8)
    ])
    
    rain transform:
        transformed = remove_reflections(rain_image, max_area=5000)
        transformed =  add_sunlight(transformed, alpha=0.8, beta=50)
        transformed = remove_fog(transformed, clipLimit=2.0, tileGridSize=16)
    """

    cv2.imwrite(f"output_manual_original.jpg", cv2.cvtColor(rain_image, cv2.COLOR_RGB2BGR))



    #rainy_image = cv2.imread(rainy_image_path)
#
    ## Get transformations
    #(image_rain_transform, image_sun_transform), segmentation_transform = get_transforms(general_augmentation=True, weather_augmentation=True)
#
    ## Apply transformations
    ## Rainy to Sunny
    #sunny_from_rain = apply_sunny_transformations(rainy_image)
    #cv2.imwrite("output_sunny_from_rain.jpg", sunny_from_rain)
#
    ## Augment Sunny Image
    #if isinstance(image_sun_transform, Compose):
    #    sunny_augmented = image_sun_transform(image=sunny_image)["image"]
    #else:
    #    sunny_augmented = image_sun_transform(sunny_image).numpy().transpose(1, 2, 0) * 255.0
    #    sunny_augmented = sunny_augmented.astype(np.uint8)
    #cv2.imwrite("output_sunny_augmented.jpg", sunny_augmented)
#
    ## Augment Rainy Image
    #rainy_augmented = image_rain_transform(image=rainy_image)["image"]
    #cv2.imwrite("output_rainy_augmented.jpg", rainy_augmented)
