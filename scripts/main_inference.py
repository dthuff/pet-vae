import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype

from dataloader import DicomDataset
from loss import KLDivergence, L2Loss
from model import VAE
from save_load import load_from_checkpoint
from train import test_loop

data_dir = '/home/daniel/datasets/ACRIN-NSCLC-FDG-PET-cleaned/'
save_dir = '../saved_models/'

# Hyper parameters:
batch_size = 64
channels = 1
img_dim = 128  # Must be factor of 16 (base VAE has 4 maxpools in encoder)
latent_dim = 256
learning_rate = 0.001
max_epochs = 500
weight_decay = 5e-7
train_val_test_split = [0.8, 0.19, 0.01]  # Proportion of data for training, validation, and testing. Sums to 1
device = 'cuda'
resume = False  # Resume training from best_epoch.tar?
use_amp = False  # Use automatic mixed precision?

# Transforms
transform_composition = Compose([
    ToTensor(),
    Resize(img_dim),
    ConvertImageDtype(torch.float)
])

# Dataset, splitting, and loaders
dataset = DicomDataset(data_dir,
                       transform=transform_composition,
                       target_transform=transform_composition)

train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                        train_val_test_split,
                                                        torch.Generator().manual_seed(91722))

val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True)

test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True)

# Load the best model and run inference on the test set
model_test = VAE(latent_dim=latent_dim,
                 img_dim=img_dim)

model_test.to(device=device)

optimizer = torch.optim.Adam(model_test.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)

model_test, optimizer, loss_dict, epoch = load_from_checkpoint(checkpoint_path=save_dir + "best_epoch.pth",
                                                               model=model_test,
                                                               optimizer=optimizer)

test_loss_kl, test_loss_recon, perf_metrics = test_loop(dataloader=test_dataloader,
                                                        model=model_test,
                                                        loss_fn_kl=KLDivergence(),
                                                        loss_fn_recon=L2Loss(),
                                                        plot_save_dir=save_dir + "test/")

# Plot PSNR and SSIM boxplots
flat_ssim = [item for sublist in perf_metrics["ssim"] for item in sublist]
flat_psnr = [item for sublist in perf_metrics["psnr"] for item in sublist]

fig, axs = plt.subplots(1, 2, figsize=(8, 6))
axs[0].hist(flat_ssim)

axs[1].hist(flat_psnr)
plt.show()
