import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BIN_COUNT = 30

def evaluate_metrics(csv_file: str):
    # Load the CSV data into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Check if 'sunny' column exists
    if 'sunny' not in df.columns:
        raise ValueError("The CSV file must have a 'sunny' column to distinguish between sunny and rainy images.")
    
    # Separate sunny and rainy images
    sunny_images = df[df['sunny'] == True]
    rainy_images = df[df['sunny'] == False]
    
    # Calculate the mean and standard deviation for sunny images
    sunny_tv_mean = sunny_images['total_variation'].mean()
    sunny_tv_std = sunny_images['total_variation'].std()
    sunny_brisque_mean = sunny_images['brisque'].mean()
    sunny_brisque_std = sunny_images['brisque'].std()
    
    # Calculate the mean and standard deviation for rainy images
    rainy_tv_mean = rainy_images['total_variation'].mean()
    rainy_tv_std = rainy_images['total_variation'].std()
    rainy_brisque_mean = rainy_images['brisque'].mean()
    rainy_brisque_std = rainy_images['brisque'].std()
    
    # Print the results
    print("Sunny Images Metrics:")
    print(f"  Total Variation - Mean: {sunny_tv_mean:.4f}, Std: {sunny_tv_std:.4f}")
    print(f"  BRISQUE - Mean: {sunny_brisque_mean:.4f}, Std: {sunny_brisque_std:.4f}")
    
    print("\nRainy Images Metrics:")
    print(f"  Total Variation - Mean: {rainy_tv_mean:.4f}, Std: {rainy_tv_std:.4f}")
    print(f"  BRISQUE - Mean: {rainy_brisque_mean:.4f}, Std: {rainy_brisque_std:.4f}")
    
    # Calculate common bin edges for both metrics
    tv_min = min(sunny_images['total_variation'].min(), rainy_images['total_variation'].min())
    tv_max = max(sunny_images['total_variation'].max(), rainy_images['total_variation'].max())
    tv_bins = np.linspace(tv_min, tv_max, BIN_COUNT)
    
    brisque_min = min(sunny_images['brisque'].min(), rainy_images['brisque'].min())
    brisque_max = max(sunny_images['brisque'].max(), rainy_images['brisque'].max())
    brisque_bins = np.linspace(brisque_min, brisque_max, BIN_COUNT)
    
    # Plotting histograms
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram for Total Variation
    axs[0].hist(sunny_images['total_variation'], bins=tv_bins, alpha=0.7, color='blue', label='Sunny')
    axs[0].hist(rainy_images['total_variation'], bins=tv_bins, alpha=0.7, color='red', label='Rainy')
    axs[0].set_title('Total Variation Distribution')
    axs[0].set_xlabel('Total Variation')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    # Histogram for BRISQUE
    axs[1].hist(sunny_images['brisque'], bins=brisque_bins, alpha=0.7, color='blue', label='Sunny')
    axs[1].hist(rainy_images['brisque'], bins=brisque_bins, alpha=0.7, color='red', label='Rainy')
    axs[1].set_title('BRISQUE Distribution')
    axs[1].set_xlabel('BRISQUE Score')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()
    
    # Save the plots
    plt.savefig('metrics_histograms.png')
    plt.show()
    
if __name__ == "__main__":
    # Specify the path to the CSV file
    csv_file_path = 'image_quality_metrics.csv'  # Adjust the path to your CSV file
    evaluate_metrics(csv_file_path)
