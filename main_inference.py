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
save_dir = './saved_models/'

train_val_test_split = [0.8, 0.1, 0.1]  # Proportion of data for training, validation, and testing. Sums to 1
device = 'cuda'
batch_size = 64
learning_rate = 0.001
max_epochs = 500
weight_decay = 5e-7

transform_composition = Compose([
    ToTensor(),
    Resize(128),
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
                             shuffle=True)

# Load the best model and run inference on the test set
model_test = VAE()
model_test.to(device=device)
optimizer = torch.optim.Adam(model_test.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)

model_test, _ = load_from_checkpoint(checkpoint_path=save_dir + "best_epoch.tar",
                                     model=model_test,
                                     optimizer=optimizer)

test_loss_kl, test_loss_recon = test_loop(dataloader=val_dataloader,
                                          model=model_test,
                                          loss_fn_kl=KLDivergence(),
                                          loss_fn_recon=L2Loss(),
                                          plot_save_dir=save_dir + "test/")
