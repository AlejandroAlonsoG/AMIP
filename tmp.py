import os
import numpy as np
from PIL import Image

# Define the class information
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

# Initialize a dictionary to store class presence counts
class_presence = {class_name: 0 for class_name, _, _ in classes}

def check_class_presence(image_path):
    """Check for the presence of each class in the image."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    for class_name, _, color in classes:
        if np.any(np.all(img_array == color, axis=-1)):
            class_presence[class_name] += 1

def process_directory(directory):
    """Process all PNG images in the given directory."""
    i = 0
    for file_name in os.listdir(directory):
        i+=1
        if file_name.endswith('.png'):
            print(f"Processing: {file_name}")
            file_path = os.path.join(directory, file_name)
            check_class_presence(file_path)
        if i > 150:
            break

if __name__ == "__main__":
    directory = "/net/ens/am4ip/datasets/project-dataset/rainy_sseg/"  # Replace with your directory path
    process_directory(directory)

    # Print the results
    for class_name, count in class_presence.items():
        print(f"Class '{class_name}' is present in {count} images.")
