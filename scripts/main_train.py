import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype

from dataloader import DicomDataset, image_transform
from loss import KLDivergence, L2Loss
from model import VAE
from plotting import plot_and_save_loss, plot_model_architecture
from save_load import save_checkpoint, load_from_checkpoint, load_config
from train import train_loop, val_loop


def parse_cl_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config")
    return arg_parser.parse_args()


if __name__ == "__main__":
    cl_args = parse_cl_args()
    config = load_config(cl_args.config)

    if not os.path.exists(config['logging']['model_save_dir']):
        os.makedirs(config['logging']['model_save_dir'])

    # Transforms
    transform_composition = image_transform(config)

    # Dataset, splitting, and loaders
    dataset = DicomDataset(config['data']['data_dir'],
                           transform=transform_composition,
                           target_transform=transform_composition)

    train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                            config['data']['train_val_test_split'],
                                                            torch.Generator().manual_seed(91722))

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config['model']['batch_size'],
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=dataset,
                                batch_size=config['model']['batch_size'],
                                shuffle=True)

    # Initialize model and optimizer
    model = VAE(latent_dim=config['model']['latent_dim'],
                img_dim=config['model']['img_dim'])
    model.to(device=config['model']['device'])

    # beta is the weight for the KL loss term. I use definition from B-VAE paper https://openreview.net/pdf?id=Sy2fzU9gl
    beta = config['model']['latent_dim'] / (config['model']['img_dim'] ** 2)

    plot_model_architecture(model=model,
                            batch_size=config['model']['batch_size'],
                            channels=config['model']['channels'],
                            img_dim=config['model']['img_dim'],
                            save_dir=config['logging']['plot_save_dir'])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['model']['learning_rate'],
                                 weight_decay=config['model']['weight_decay'])

    # If we are resuming training, load the state dict for the model and the optimizer from best_epoch
    if config['model']['resume']:
        best_model = os.path.join(config['logging']['model_save_dir'], "best_epoch.pth")
        model, optimizer, loss_dict, start_epoch = load_from_checkpoint(checkpoint_path=best_model,
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
        best_val_loss = 1e12
        best_val_epoch = 0

    # Training loop
    for t in range(start_epoch, config['model']['max_epochs']):
        print(f"Epoch {t}\n-------------------------------")
        train_loss_kl, train_loss_recon = train_loop(dataloader=train_dataloader,
                                                     model=model,
                                                     loss_fn_kl=KLDivergence(),
                                                     loss_fn_recon=L2Loss(),
                                                     beta=beta,
                                                     optimizer=optimizer,
                                                     amp_on=config['model']['use_amp'])

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
                           save_dir=config['logging']['model_save_dir'])

        # Save a checkpoint every 10 epochs
        if t % config['logging']['save_model_every_n_epochs'] == 0:
            save_checkpoint(save_path=os.path.join(config['logging']['model_save_dir'], f"epoch_{t}.pth"),
                            model=model,
                            optimizer=optimizer,
                            loss_dict=loss_dict,
                            epoch_number=t)

        # If this epoch has the best validation loss, save it to "best_epoch.tar"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = t
            save_checkpoint(save_path=os.path.join(config['logging']['model_save_dir'], "best_epoch.pth"),
                            model=model,
                            optimizer=optimizer,
                            loss_dict=loss_dict,
                            epoch_number=t)

    print("Done training.")
    print("Best epoch was: " + str(best_val_epoch))
