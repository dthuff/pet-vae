import os
import matplotlib.pyplot as plt
import numpy as np


def plot_examples(X, y_pred, plot_path):
    """

    Args:
        X (tensor) : model inputs
        y_pred (tensor) : model reconstruction
        plot_path (string) : path to folder to save images

    Returns:

    """
    # Limit plot to 32 slices -
    X = X[:32, :, :, :]
    y_pred = y_pred[:32, :, :, :]

    # Plot input slices X and predicted slices y_pred
    fig, axs = plt.subplots(nrows=int(X.shape[0] / 4),
                            ncols=8,
                            figsize=(8, int(X.shape[0] / 4)))
    axs = axs.flatten()

    for i, (img, recon_img) in enumerate(zip(X, y_pred)):
        axs[2 * i].imshow(np.squeeze(np.asarray(img)), cmap='inferno')
        axs[(2 * i) + 1].imshow(np.squeeze(np.asarray(recon_img)), cmap='inferno')

    # Hide axes and whitespace
    for a in axs:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    # Create the save_dir if it does not exist
    save_dir, _ = os.path.split(plot_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print("Saved example image to: " + plot_path)


def plot_and_save_loss(loss_dict, save_dir):
    """

    Args:
        loss_dict (dictionary) : dictionary containing lists of loss values per epoch
        save_dir (string) : path to save directory

    Returns:

    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    for loss in loss_dict.values():
        ax.plot(loss)
    ax.legend(loss_dict.keys())
    plt.savefig(save_dir + "loss.png", dpi=150)
    plt.close(fig)
