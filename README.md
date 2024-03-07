![alt text](plots/all_tiles.gif "Axial PET slices. Odd columns: input, Even columns: output")

# pet-vae
2D variational autoencoder for reconstructing PET images implemented in pytorch

Trained using data from [ACRIN-NSCLC-FDG-PET](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=39879162) hosted by The Cancer Imaging Archive.

Converges in ~3-5 hours training on an NVIDIA 1080 (8 GB).

# Installation

### Clone the repository

    git clone https://github.com/dthuff/pet-vae.git

### Install dependencies with [Poetry](https://python-poetry.org/):

    cd pet-vae
    poetry install --no-root

# Usage

### Run training and inference via:
    
Training:

    poetry run python main_training.py --config /path/to/my/training_config.yml

Inference: (requires that you point to a saved model `.pth` in config

    poetry run python main_inference.py --config /path/to/my/test_config.yml

Example config is provided: `configs/train_config.yml`

# Troubleshooting

### Out of memory? Try:

* Reducing config.model.img_dim
* Reducing config.model.latent_dim
* Reducing number of encoder/decoder blocks, or number of filters per block in `model.py`

### Model not converging? Try:

* Adjusting `beta` in `train.py`. `beta` controls the weight of the KL-loss relative to the Reconstruction loss. Check your loss.png to see which loss term is dominant and adjust `beta` accordingly.
* Checking data consistency/correctness
* Acquiring additional data

### Loss diverging (infs or nans)? Try:

* Increasing config.model.latent_dim
