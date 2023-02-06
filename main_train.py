import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype

from dataloader import DicomDataset
from model import VAE
from loss import KLDivergence, L2Loss
from train import train_loop, val_loop, test_loop
from save_load import save_checkpoint, load_from_checkpoint
from plotting import plot_and_save_loss

# Hyper parameters:
batch_size = 64
learning_rate = 0.001
max_epochs = 300
weight_decay = 5e-7
train_val_test_split = [0.8, 0.1, 0.1]  # Proportion of data for training, validation, and testing. Sums to 1
device = 'cuda'
resume = False  # Resume training from best_epoch.tar?

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
                                                        train_val_test_split,
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

# Initialize model and optimizer
model = VAE()
model.to(device=device)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)

# If we are resuming training, load the state dict for the model and the optimizer from best_epoch
if resume:
    model, optimizer = load_from_checkpoint(checkpoint_path=save_dir + "best_epoch.tar",
                                            model=model,
                                            optimizer=optimizer)

# Training loop
best_val_loss = 10000000000000
best_val_epoch = 0

# Initialize lists for saving losses at each epoch
loss_dict = {"TRAIN_LOSS_KL": [],
             "TRAIN_LOSS_RECON": [],
             "VAL_LOSS_KL": [],
             "VAL_LOSS_RECON": []
             }

for t in range(max_epochs):
    print(f"Epoch {t}\n-------------------------------")
    train_loss_kl, train_loss_recon = train_loop(dataloader=train_dataloader,
                                                 model=model,
                                                 loss_fn_kl=KLDivergence(),
                                                 loss_fn_recon=L2Loss(),
                                                 optimizer=optimizer,
                                                 amp_on=False)

    val_loss_kl, val_loss_recon = val_loop(dataloader=val_dataloader,
                                           model=model,
                                           loss_fn_kl=KLDivergence(),
                                           loss_fn_recon=L2Loss(),
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

    # TODO - plot loss over time and save
    plot_and_save_loss(loss_dict=loss_dict,
                       save_dir=save_dir)

    # Save a checkpoint every 5 epochs
    if t % 5 == 0:
        save_checkpoint(save_path=save_dir + "epoch_" + str(t) + ".tar",
                        model=model,
                        optimizer=optimizer,
                        loss_kl=val_loss_kl,
                        loss_recon=val_loss_recon,
                        epoch_number=t)

    # If this epoch has the best validation loss, save it to "best_epoch.tar"
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = t
        save_checkpoint(save_path=save_dir + "best_epoch.tar",
                        model=model,
                        optimizer=optimizer,
                        loss_kl=val_loss_kl,
                        loss_recon=val_loss_recon,
                        epoch_number=t)

print("Done training.")
print("Best epoch was: " + str(best_val_epoch))

# Load the best model and run inference on the test set
model_test = VAE()
model_test.to(device=device)
model_test, _ = load_from_checkpoint(checkpoint_path=save_dir + "best_epoch.tar")

test_loop(dataloader=test_dataloader,
          model=model_test,
          loss_fn_kl=KLDivergence(),
          loss_fn_recon=L2Loss(),
          plot_save_dir=save_dir + "test/")
