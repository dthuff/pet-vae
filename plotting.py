import os

import matplotlib.pyplot as plt
import torch
from torchviz import make_dot


def plot_examples(X, y_pred, plot_path):
    """

    Args:
        X (tensor) : model input
        y_pred (tensor) : model reconstruction
        plot_path (string) : path to folder to save images

    Returns:
        None
    """
    # Limit plot to 32 slices
    if X.shape[0] > 32:
        X = X[:32, :, :, :]
        y_pred = y_pred[:32, :, :, :]

    # Plot input slices X and predicted slices y_pred
    fig, axs = plt.subplots(nrows=int(X.shape[0] / 4),
                            ncols=8,
                            figsize=(8, int(X.shape[0] / 4)))
    axs = axs.flatten()

    for i, (img, recon_img) in enumerate(zip(X, y_pred)):
        img = img.cpu().detach().squeeze().numpy()
        recon_img = recon_img.cpu().detach().squeeze().numpy()
        axs[2 * i].imshow(img, cmap='inferno', vmin=0, vmax=img.max())
        axs[(2 * i) + 1].imshow(recon_img, cmap='inferno', vmin=0, vmax=img.max())

    for a in axs:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    save_dir, _ = os.path.split(plot_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(plot_path, dpi=150)
    plt.close(fig)


def plot_and_save_loss(loss_dict, save_dir):
    """

    Args:
        loss_dict (dictionary) : dictionary containing lists of loss values per epoch
        save_dir (string) : path to save directory

    Returns:

    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax2 = ax.twinx()

    ax.plot(loss_dict["TRAIN_LOSS_KL"], c='blue')
    ax.plot(loss_dict["VAL_LOSS_KL"], c='cornflowerblue')
    ax2.plot(loss_dict["TRAIN_LOSS_RECON"], c='red')
    ax2.plot(loss_dict["VAL_LOSS_RECON"], c='lightcoral')

    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_ylabel("KL Loss")
    ax2.set_ylabel("Recon Loss")
    ax.set_xlabel("Epoch")
    fig.legend(["KL loss (train)", "KL loss (val)", "Recon loss (train)", "Recon loss (val)"],
               bbox_to_anchor=(0.9, 0.85))
    plt.savefig(os.path.join(save_dir, "loss.png"), dpi=150)
    plt.close(fig)


def plot_model_architecture(model, batch_size, channels, img_dim, save_dir):
    x = torch.randn(batch_size, channels, img_dim, img_dim)
    x = x.to(device="cuda")
    y = model(x)
    make_dot(y, params=dict(model.named_parameters())).render(os.path.join(save_dir, "model.png"))


def plot_performance(ssim, psnr, save_dir):
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    axs[0].hist(ssim)
    axs[0].set_title("SSIM")
    axs[1].hist(psnr)
    axs[1].set_title("PSNR")
    axs[1].set_xlabel("dB")
    plt.savefig(os.path.join(save_dir, "perf_metrics.png"))
    print(f"Saved performance metric plot to: {os.path.join(save_dir, 'perf_metrics.png')}")
