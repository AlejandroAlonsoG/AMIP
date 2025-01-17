# Project Title

## Introduction
This project, developed as part of the "Advanced Methods for Image Processing" course, utilizes a UNet-based approach for multiclass semantic segmentation of driving images. The dataset includes scenarios captured in both rainy and sunny weather conditions, presenting unique challenges for segmentation.

## How to Run
You can evaluate the model using the provided script:

```bash
python3 ./other_files/model_evaluation.py
```

The best-performing model is stored as `best_model.pth`, corresponding exclusively to augmentations applied for sunny and rainy conditions.

## Requirements
Ensure you have all dependencies installed before running the scripts. You can install them using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Code Organization

- **main.py**:
  - Entry point of the code.
  - Fully commented and straightforward to follow.
  - Logs outputs to both the terminal and a log file to ensure outputs are preserved even if the terminal is closed.
  - **Usage:**
    ```bash
    python3 main.py <config_file>
    ```
    Config files are stored in the `./configs/` directory. These files centralize parameters for flexibility and easier execution of multiple runs. This setup simplifies parameter sharing across files and facilitates scripting for batch execution in environments like CREMI.

- **datasets.py**:
  - Handles dataset reading and allows splitting datasets into distinct groups to prevent data leakage.
  - Supports augmentations with specified probabilities.

- **transforms_loaders.py**:
  - Contains transformations for data augmentation, including sunny-to-rainy and rainy-to-sunny conversions.
  - Also houses standard transformations for data preprocessing.

- **general_loader**:
  - Loads commonly required components, enabling easy addition of different losses through the config file.

- **UNet Implementation**:
  - Manually constructed based on a reference diagram.
  - Facilitates semantic segmentation with raw logits for each class.

- **Trainer**:
  - Logs metrics per epoch and provides evaluation functionality for the test set.
  - Current metrics include Mean IoU and Weighted IoU (weighted by class frequency).

- **execute_cremi.py**:
  - Script for running tasks on CREMI.
 
- **class_distribution.py**:
  - calculate class distribution proportions and create bar chart figure 

- **metrics.py**:
  - contains quantitative image quality metric definitions and a calculation on both sunny and rainy images, create a .csv file with the results

- **metrics_evaluation.py**:
  - evaluate the .csv file created by metrics.py, create histograms of metrics

## Notes
- Augmentation strategies were implemented to simulate transitions between sunny and rainy conditions using the Albumentations library. Specific transformations were applied for each condition to improve generalization.
- Metrics are logged to monitor training progress and evaluate model performance across different classes and scenarios.

