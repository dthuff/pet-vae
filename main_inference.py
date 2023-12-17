import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataloader import DicomDataset, image_transform
from loss import KLDivergence, L2Loss
from model import VAE
from save_load import load_from_checkpoint, load_config, create_output_directories
from main_train import parse_cl_args
from train import test_loop

if __name__ == "__main__":
    cl_args = parse_cl_args()
    config = load_config(cl_args.config)

    create_output_directories(config)

    transform_composition = image_transform(config)

    dataset = DicomDataset(config['data']['data_dir'],
                           transform=transform_composition,
                           target_transform=transform_composition)

    train_dataset, val_dataset, test_dataset = random_split(dataset,
                                                            config['data']['train_val_test_split'],
                                                            torch.Generator().manual_seed(91722))

    val_dataloader = DataLoader(val_dataset,
                                batch_size=config['model']['batch_size'],
                                shuffle=True)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config['model']['batch_size'],
                                 shuffle=True,
                                 drop_last=True)

    model_test = VAE(latent_dim=config['model']['latent_dim'],
                     img_dim=config['model']['img_dim'])

    model_test.to(device=config['model']['device'])

    optimizer = torch.optim.Adam(model_test.parameters(),
                                 lr=config['model']['learning_rate'],
                                 weight_decay=config['model']['weight_decay'])

    model_test, optimizer, loss_dict, epoch = load_from_checkpoint(
        checkpoint_path=os.path.join(config['logging']['model_save_dir'], "best_epoch.pth"),
        model=model_test,
        optimizer=optimizer)

    test_loss_kl, test_loss_recon, perf_metrics = test_loop(dataloader=test_dataloader,
                                                            model=model_test,
                                                            loss_fn_kl=KLDivergence(),
                                                            loss_fn_recon=L2Loss(),
                                                            plot_save_dir=config['logging']['plot_save_dir'], )

    # Plot PSNR and SSIM boxplots
    flat_ssim = [item for sublist in perf_metrics["ssim"] for item in sublist]
    flat_psnr = [item for sublist in perf_metrics["psnr"] for item in sublist]

    # TODO - move to plotting.py
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    axs[0].hist(flat_ssim)
    axs[1].hist(flat_psnr)
    plt.savefig(os.path.join(config["logging"]["plot_save_dir"], "perf_metrics.png"))
    plt.show()
