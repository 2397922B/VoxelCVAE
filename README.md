# Conditional Variational Autoencoder for Single View Reconstruction

## Overview

This project implements a Conditional Variational Autoencoder (CVAE) for reconstructing 3D shapes from a single 2D image. The approach leverages a generative voxel-based method that enables the generation of diverse and detailed 3D reconstructions, even in the presence of occlusions.
<img width="685" alt="Screenshot 2024-07-08 at 12 14 02 AM" src="https://github.com/2397922B/VoxelCVAE/assets/97679718/af37b836-e53f-45a6-bcdc-83caa42294c6">

## Installation

### Prerequisites

- Python 3.6+
- Git

### Steps

1. **Clone the repository:**

2. **Install required Python packages:**
   ```sh
   pip install -r requirements.txt
   ```

## Dataset

### Download the Training and Testing Datasets

- [Training Dataset](https://drive.google.com/file/d/1f0Zn_VvHBOxnpZm_SXo10zQZMyNWQ8EJ/view?usp=share_link)
- [Testing Dataset](https://drive.google.com/file/d/1S44ZrYM32PbiotbTP3InoliiaggHOmBI/view?usp=share_link)

## Pretrained Weights

Pretrained weights without triplet loss can be downloaded here:

- [VAE Weights](https://drive.google.com/file/d/1tPqSg51ArQuWdVvfuLAhPrHggSBHdosi/view?usp=share_link)
- [CNN Weights](https://drive.google.com/file/d/1GySiPu9tOYuXXNF9E2ne5a08_0BzlceS/view?usp=share_link)

## Training the Model

1. **Update `train.py` with the correct data path:**
   ```python
   data_train = NpyTarDataset('/path/to/tar')
   ```

2. **Navigate to the VAE directory:**
   ```sh
   cd VAE
   ```

3. **Run the training script:**
   ```sh
   python train.py
   ```

Weights for both VAE and CNN will be saved within the VAE directory.

## Testing the Model

1. **Update `test.py` with the correct data paths:**
   ```python
   data_test = NpyTarDataset('/home/conorbrown/Downloads/VAE/datasets/legs_test.tar')
   data_train = NpyTarDataset('/home/conorbrown/Downloads/VAE/datasets/legs_train.tar')
   
   path = "path/to/vae_weights"
   path2 = "path/to/cnn_weights"
   ```

2. **Run the testing script:**
   ```sh
   python test.py
   ```

Results will be saved under `VAE/reconstructions`.

## Results

The results of the model's performance will be saved in the `VAE/reconstructions` directory.

## Contribution

Feel free to contribute to this project by submitting a pull request or opening an issue.

## License

[MIT License](LICENSE)

