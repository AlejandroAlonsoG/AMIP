import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define class mappings
class_colors = {
    "Unlabeled": (0, 0, 0),
    "Dynamic": (111, 74, 0),
    "Ground": (81, 0, 81),
    "Road": (128, 64, 128),
    "Sidewalk": (244, 35, 232),
    "Parking": (250, 170, 160),
    "Rail track": (230, 150, 140),
    "Building": (70, 70, 70),
    "Wall": (102, 102, 156),
    "Fence": (190, 153, 153),
    "Guard rail": (180, 165, 180),
    "Bridge": (150, 100, 100),
    "Tunnel": (150, 120, 90),
    "Pole or pole group": (153, 153, 153),
    "Traffic light": (250, 170, 30),
    "Traffic sign": (220, 220, 0),
    "Vegetation": (107, 142, 35),
    "Ground": (152, 251, 152),
    "Sky": (70, 130, 180),
    "Person": (220, 20, 60),
    "Rider": (255, 0, 0),
    "Truck": (0, 0, 70),
    "Bus": (0, 60, 100),
    "Caravan": (0, 0, 90),
    "Trailer": (0, 0, 110),
    "Train": (0, 80, 100),
    "Motorcycle": (0, 0, 230),
    "Bicycle": (119, 11, 32)
}

# Initialize counters for each class
def initialize_class_counter():
    return {class_name: 0 for class_name in class_colors.keys()}

def count_classes_in_folder(folder_path, class_colors):
    class_counter = initialize_class_counter()
    num_frames = 0
    for file_name in os.listdir(folder_path):
        num_frames += 1
        
        file_path = os.path.join(folder_path, file_name)
        
        # Load image
        img = Image.open(file_path)
        img = img.convert('RGB')
        img = np.array(img)
    
        for class_name, color in class_colors.items():
            mask = (img[:, :, 0] == color[0]) & (img[:, :, 1] == color[1]) & (img[:, :, 2] == color[2])
            if np.any(mask):
                class_counter[class_name] += 1
    
    return class_counter, num_frames

def plot_class_distribution(rainy_counts, sunny_counts):
    classes = list(class_colors.keys())
    
    # Extract values for plotting
    rainy_values = [rainy_counts[class_name] for class_name in classes]
    sunny_values = [sunny_counts[class_name] for class_name in classes]
    
    x = np.arange(len(classes))
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    axs[0].bar(x, rainy_values, color="blue", alpha=0.7)
    axs[0].set_title("Class Distribution in Rainy Images")
    axs[0].set_ylabel("Number of Appearances")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(classes, rotation=90)

    axs[1].bar(x, sunny_values, color="orange", alpha=0.7)
    axs[1].set_title("Class Distribution in Sunny Images")
    axs[1].set_ylabel("Number of Appearances")
    axs[1].set_xlabel("Class")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(classes, rotation=90)
    
    plt.tight_layout()
    plt.savefig("class_distributions.png")
    plt.show()

def print_class_counts(rainy_counts, rainy_frames, sunny_counts, sunny_frames):
    """
    Print the class names and their appearance counts for rainy and sunny datasets.
    
    Args:
        rainy_counts (dict): Class counts for rainy images.
        sunny_counts (dict): Class counts for sunny images.
    """
    print(f"{'Class':<20}{'Rainy Count':<15}{'Sunny Count':<15}")
    for class_name in class_colors.keys():
        rainy_prop = rainy_counts[class_name] / rainy_frames
        sunny_prop = sunny_counts[class_name] / sunny_frames
        print(f"{class_name:<20}{sunny_prop:<15}{rainy_prop:<15}")
        
if __name__ == "__main__":
    rainy_folder = "/net/ens/am4ip/datasets/project-dataset/rainy_sseg"
    sunny_folder = "/net/ens/am4ip/datasets/project-dataset/sunny_sseg"

    print("Counting classes in rainy images...")
    rainy_counts, rainy_frames = count_classes_in_folder(rainy_folder, class_colors)
    
    print("Counting classes in sunny images...")
    sunny_counts, sunny_frames = count_classes_in_folder(sunny_folder, class_colors)

    print_class_counts(rainy_counts, rainy_frames, sunny_counts, sunny_frames)
    
    print("Plotting class distributions...")
    plot_class_distribution(rainy_counts, sunny_counts)
