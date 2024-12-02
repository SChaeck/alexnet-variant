# CIFAR-10 Image Classification Project

This repository contains code for classifying CIFAR-10 images using various models and analyzing the results through confusion matrices and embedding space visualization.

## Directory Structure

- **`data/`**  
  Contains data loading and preprocessing scripts:
  - `data_load.py`: Handles data loading and transformations.
  - `data_visualize_save()`: Saves visualizations of raw and preprocessed images.

- **`model/`**  
  Includes model definitions:
  - `alexnet.py`: Standard AlexNet implementation.
  - `alexnet_with_residual.py`: AlexNet with residual connections (AlexNet-R).
  - `alexnet_with_classifier.py`: AlexNet with an external classifier (AlexNet-C).
  - `pca.py`: PCA model for reducing embedding dimensions.

- **`main/`**  
  Contains scripts for training, testing, and analysis:
  - `training.py`: Defines the training workflow.
  - `test.py`: Generates confusion matrices for each model.
  - `embedding_space_analysis.py`: Visualizes the embedding space of models.

- **`outputs/`**  
  Stores output results:
  - `outputs/{model}/`: Model parameters and logs for each model.
  - `outputs/pictures/`: Visualizations including confusion matrices and embedding space analysis.

- **`run_code.sh`**  
  A shell script to automate the execution process.

## How to Run

1. Grant execution permission to the script:
   ```bash
   chmod +x run_code.sh
   ```
2. Execute the script:
   ```bash
   ./run_code.sh
   ```
### Expected Outputs

- **Model Parameters and Logs**:  
  Saved in `outputs/{model}/`. This includes:
  - Training parameters.
  - The best-performing model's parameters.

- **Visualizations**:  
  Saved in `outputs/pictures/`. This includes:
  - Preprocessed data visualizations.
  - Confusion matrices for each model.
  - Embedding space visualizations.

If you have any questions or encounter issues, feel free to reach out!
