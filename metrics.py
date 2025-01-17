# Define image quality metric functions 
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2

import scipy.stats
from scipy.ndimage import gaussian_filter
import scipy.stats as stats

from typing import Tuple

import transforms_loader as tl

class TotalVariation:
    @staticmethod
    def calculate(image: np.ndarray) -> float:
        """
        Calculate the Total Variation of an image using the calc_tv function.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            float: Total Variation of the image.
        """
        # Convert image to PyTorch tensor
        img_tensor = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        
        # Handle each channel separately
        tv_per_channel = []
        for channel in img_tensor[0]:  # Iterate over RGB channels
            channel = channel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            
            # Define gradient filters
            grad_filters = torch.tensor([[0, 0, 0],
                                         [0, -1, 1],
                                         [0, 0, 0]], dtype=channel.dtype, device=channel.device)
            grad_filters = torch.concat([grad_filters[None, None, ...],  # Horizontal gradient
                                         grad_filters.T[None, None, ...]], dim=0)  # Vertical gradient
            
            # Compute gradients
            grad = torch.conv2d(channel, grad_filters, padding=0)  # 1 x 2 x H x W
            
            # Compute gradient magnitude
            gm = torch.sqrt(torch.sum(grad**2, dim=1))  # 1 x H x W
            tv_per_channel.append(torch.sum(gm).cpu().numpy())  # Total variation for this channel
        
        # Average total variation across all channels
        total_variation = sum(tv_per_channel) / len(tv_per_channel)
        return total_variation

class BRISQUE:
    @staticmethod
    def calculate(image: np.ndarray) -> float:
        """
        Calculate the BRISQUE score for a given image.

        Args:
            image (np.ndarray): Input grayscale image.

        Returns:
            float: BRISQUE score of the image.
        """
        # Step 1: Convert image to MSCN coefficients
        mscn_coefficients = BRISQUE.compute_mscn_coefficients(image)
        
        # Step 2: Extract BRISQUE features
        brisque_features = BRISQUE.extract_features(mscn_coefficients)
        
        # Step 3: Compute BRISQUE score (distance metric, simplified for demonstration)
        # Replace this with the model or statistical comparison used for scoring
        brisque_score = np.sum(brisque_features)  # Dummy scoring, replace with actual
        return brisque_score

    @staticmethod
    def compute_mscn_coefficients(image: np.ndarray) -> np.ndarray:
        """
        Compute the MSCN coefficients for the image.

        Args:
            image (np.ndarray): Input grayscale image.

        Returns:
            np.ndarray: MSCN coefficients.
        """
        mu = gaussian_filter(image, sigma=7)
        sigma = np.sqrt(gaussian_filter(image ** 2, sigma=7) - mu ** 2 + 1e-8)
        mscn_coefficients = (image - mu) / sigma
        return mscn_coefficients

    @staticmethod
    def fit_ggd(data: np.ndarray) -> Tuple[float, float]:
        """
        Fit a Generalized Gaussian Distribution (GGD) to the data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            Tuple[float, float]: Fitted alpha and sigma parameters.
        """
        gamma_range = np.arange(0.2, 10.0, 0.001)
        gamma_function = lambda x: (stats.gamma(1.0 / x).pdf(1) * stats.gamma(3.0 / x).pdf(1)) / (stats.gamma(2.0 / x).pdf(1) ** 2)

        # Estimate alpha
        sigma_sq = np.mean(data ** 2)
        E = np.mean(np.abs(data))
        rho = sigma_sq / (E ** 2)
        alpha = gamma_range[np.argmin(np.abs(gamma_function(gamma_range) - rho))]

        # Estimate sigma
        sigma = np.sqrt(sigma_sq)
        return float(alpha), float(sigma)

    @staticmethod
    def extract_features(mscn_coefficients: np.ndarray) -> np.ndarray:
        """
        Extract BRISQUE features from MSCN coefficients.

        Args:
            mscn_coefficients (np.ndarray): MSCN coefficients of the image.

        Returns:
            np.ndarray: BRISQUE features as a flat array.
        """
        features = []

        # Extract GGD parameters for MSCN coefficients
        alpha, sigma = BRISQUE.fit_ggd(mscn_coefficients)
        features.append(alpha)
        features.append(sigma)

        # Compute pairwise products in horizontal, vertical, diagonal, and anti-diagonal directions
        shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for shift in shifts:
            shifted_mscn = np.roll(mscn_coefficients, shift, axis=(0, 1))
            pairwise_product = mscn_coefficients * shifted_mscn
            alpha, sigma = BRISQUE.fit_ggd(pairwise_product)
            features.append(alpha)
            features.append(sigma)

        # Ensure all features are scalars
        features = [float(f) for f in features]

        return np.array(features)

def calculate_metrics(dataset_dir, output_file):
    """
    Calculate Total Variation and BRISQUE for each image in the dataset.

    Args:
        dataset_dir (str): Path to the dataset root directory.
        output_file (str): Path to save the output DataFrame as a CSV file.
    """
    results = []
    for mode in ['sunny', 'rainy']:
        image_dir = os.path.join(dataset_dir, f"{mode}_images")
        idx = 1
        max_idx = len(os.listdir(image_dir))
        for img_file in os.listdir(image_dir):
            if idx <= 0:
                idx += 1
                continue
            
            img_path = os.path.join(image_dir, img_file)
            try:
                # Load the image
                img = Image.open(img_path).convert("RGB")
                img_array = np.array(img)
                
                if_sunny = True if mode == 'sunny' else False
                
                # generate augmentations to test effent on metrics
                if if_sunny:
                    augmentation = tl.get_sunny_augmentation()
                else:
                    augmentation = tl.get_rainy_augmentation()
                
                augmented_image1 = augmentation(img_array)
                augmented_image2 = augmentation(img_array)
                augmented_image3 = augmentation(img_array)
                
                # Calculate metrics
                tv = TotalVariation.calculate(img_array)
                br = BRISQUE.calculate(img_array)
                tv1 = TotalVariation.calculate(augmented_image1)
                br1 = BRISQUE.calculate(augmented_image1)
                tv2 = TotalVariation.calculate(augmented_image2)
                br2 = BRISQUE.calculate(augmented_image2)
                tv3 = TotalVariation.calculate(augmented_image3)
                br3 = BRISQUE.calculate(augmented_image3)
                augmented_tv =  (tv1 + tv2 + tv3) / 3
                augmented_br = (br1 + br2 + br3) / 3
                
                # Store results
                results.append({
                    'image_name': img_file,
                    'sunny': if_sunny,
                    'total_variation': tv,
                    'brisque': br,
                    'augmented_total_variation': augmented_tv,
                    'augmented_brisque': augmented_br
                })
                
                # Print results
                print(f'idx: {idx}/{max_idx} - image_name: {img_file}, sunny: {if_sunny}, tv: {tv}, brisque: {br}, augmented tv: {augmented_tv}, augmented br: {augmented_br}')

            except Exception as e:
               print(f"Error processing {img_file}: {e}")

            idx += 1
            if idx > 50:
                break
            
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


# Example usage
dataset_dir = "/net/ens/am4ip/datasets/project-dataset"
output_file = 'image_quality_metrics_augmentation.csv'
calculate_metrics(dataset_dir, output_file)
