# pet-vae
2D variational autoencoder for reconstructing PET images implemented in pytorch

Trained using data from [ACRIN-NSCLC-FDG-PET](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=39879162) hosted by The Cancer Imaging Archive.

Converges in ~3-5 hours training on a NVIDIA 1080 (8 GB).

# Installation

Clone the repository

    git clone https://github.com/dthuff/pet-vae.git

Install dependencies with [Poetry](https://python-poetry.org/):

    cd pet-vae
    poetry install --no-root

# Usage

Run training and inference via:
    
Training:

    poetry run python main_training.py --config /path/to/my/training_config.yml

Inference: (requires that you point to a saved model `.pth` in config

    poetry run python main_inference.py --config /path/to/my/test_config.yml

Example config is provided: `configs/train_config.yml`

