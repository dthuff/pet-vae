import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype

from dataloader import DicomDataset
from loss import KLDivergence, L2Loss
from model import VAE
from plotting import plot_and_save_loss, plot_model_architecture
from save_load import save_checkpoint, load_from_checkpoint
from train import train_loop, val_loop

# Hyper parameters:
batch_size = 64
channels = 1
img_dim = 128  # Must be factor of 16 (base VAE has 4 maxpools in encoder)
latent_dim = 256
beta = latent_dim / (img_dim ** 2)  # Weight for KL loss term. From B-VAE paper https://openreview.net/pdf?id=Sy2fzU9gl
learning_rate = 0.001
max_epochs = 2000
weight_decay = 5e-7
train_val_test_split = [0.8, 0.1, 0.1]  # Proportion of data for training, validation, and testing. Sums to 1
device = 'cuda'
resume = False  # Resume training from best_epoch.tar?
use_amp = False  # Use automatic mixed precision?

# Paths
data_dir = '/home/daniel/datasets/ACRIN-NSCLC-FDG-PET-cleaned/'
save_dir = '../saved_models/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

val_dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True)

# Initialize model and optimizer
model = VAE(latent_dim=latent_dim,
            img_dim=img_dim)
model.to(device=device)

plot_model_architecture(model=model,
                        batch_size=batch_size,
                        channels=channels,
                        img_dim=img_dim,
                        save_dir=save_dir)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)

# If we are resuming training, load the state dict for the model and the optimizer from best_epoch
if resume:
    model, optimizer, loss_dict, start_epoch = load_from_checkpoint(checkpoint_path=save_dir + "best_epoch.pth",
                                                                    model=model,
                                                                    optimizer=optimizer)
    best_val_loss = min(loss_dict["VAL_LOSS_RECON"])
    best_val_epoch = np.argmin(loss_dict["VAL_LOSS_RECON"])
else:
    start_epoch = 0
    loss_dict = {"TRAIN_LOSS_KL": [],
                 "TRAIN_LOSS_RECON": [],
                 "VAL_LOSS_KL": [],
                 "VAL_LOSS_RECON": []
                 }
    best_val_loss = 10000000000000
    best_val_epoch = 0

# Training loop
for t in range(start_epoch, max_epochs):
    print(f"Epoch {t}\n-------------------------------")
    train_loss_kl, train_loss_recon = train_loop(dataloader=train_dataloader,
                                                 model=model,
                                                 loss_fn_kl=KLDivergence(),
                                                 loss_fn_recon=L2Loss(),
                                                 beta=beta,
                                                 optimizer=optimizer,
                                                 amp_on=use_amp)

    val_loss_kl, val_loss_recon = val_loop(dataloader=val_dataloader,
                                           model=model,
                                           loss_fn_kl=KLDivergence(),
                                           loss_fn_recon=L2Loss(),
                                           beta=beta,
                                           epoch_number=t)

    val_loss = val_loss_kl + val_loss_recon
    print(f"Validation loss for epoch {t:>2d}:")
    print(f"    KL loss:   {val_loss_kl:>15.2f}")
    print(f"    Recon loss:{val_loss_recon:>15.2f}")

    # Append losses for this epoch
    loss_dict["TRAIN_LOSS_KL"].append(train_loss_kl)
    loss_dict["TRAIN_LOSS_RECON"].append(train_loss_recon)
    loss_dict["VAL_LOSS_KL"].append(val_loss_kl)
    loss_dict["VAL_LOSS_RECON"].append(val_loss_recon)

    plot_and_save_loss(loss_dict=loss_dict,
                       save_dir=save_dir)

    # Save a checkpoint every 10 epochs
    if t % 50 == 0:
        save_checkpoint(save_path=save_dir + "epoch_" + str(t) + ".pth",
                        model=model,
                        optimizer=optimizer,
                        loss_dict=loss_dict,
                        epoch_number=t)

    # If this epoch has the best validation loss, save it to "best_epoch.tar"
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = t
        save_checkpoint(save_path=save_dir + "best_epoch.pth",
                        model=model,
                        optimizer=optimizer,
                        loss_dict=loss_dict,
                        epoch_number=t)

print("Done training.")
print("Best epoch was: " + str(best_val_epoch))
