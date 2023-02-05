import torch
from dataloader import DicomDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype
from model import VAE
from loss import KLDivergence, L2Loss
from train import train_loop, val_loop
from save_load import save_checkpoint, load_from_checkpoint
from torch.utils.data import random_split
#import matplotlib.pyplot as plt

# Hyperparameters:
batch_size = 64
learning_rate = 0.001
max_epochs = 100
weight_decay = 5e-7
device = 'cuda'
data_dir = '/home/daniel/Projects/pet-vae/data/ACRIN-NSCLC-FDG-PET-cleaned/'
save_dir = './saved_models/'

# Transforms
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
                                                        [0.7, 0.1, 0.2], 
                                                        torch.Generator().manual_seed(91722))

train_dataloader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True)

val_dataloader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=True)

test_dataloader = DataLoader(test_dataset, 
                                batch_size=batch_size, 
                                shuffle=True)

# Initilize model and optimizer
model = VAE()
model.to(device=device)
optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate, 
                                weight_decay=weight_decay)

# Training loop
best_val_loss = 10000000000000

for t in range(max_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloader=train_dataloader, 
                model=model, 
                loss_fn_KL=KLDivergence(), 
                loss_fn_recon=L2Loss(), 
                optimizer=optimizer,
                amp_on=False)
    
    loss_KL, loss_recon = val_loop(dataloader=val_dataloader,
                                    model=model,
                                    loss_fn_KL=KLDivergence(), 
                                    loss_fn_recon=L2Loss())
    val_loss = loss_KL + loss_recon
    
    # Save a checkpoint every 5 epochs
    if t % 5 == 0:
        save_checkpoint(save_path=save_dir + "epoch_" + str(t) + ".tar",
                        model=model,
                        optimizer=optimizer,
                        loss_KL=loss_KL,
                        loss_recon=loss_recon,
                        epoch_number=t)
    
    # If this epoch has the best validation loss, save it to "best_epoch.tar"
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(save_path=save_dir + "best_epoch.tar",
                        model=model,
                        optimizer=optimizer,
                        loss_KL=loss_KL,
                        loss_recon=loss_recon,
                        epoch_number=t)
        


# Load the best model and run inference on the test set
model_test = VAE()
optimizer_test = torch.optim.Adam()

model_test, _ = load_from_checkpoint(checkpoint_path=save_dir + "best_epoch.tar")

test_loop()