import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_examples(X, y_pred, epoch_number):

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

    plt.savefig("./saved_models/validation_images_epoch_" + str(epoch_number) + ".png", dpi=150)
