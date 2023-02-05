import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_examples(X, y_pred):
    fig, axs = plt.subplots(int(X.shape[0] / 4), 8)
    axs = axs.flatten()

    for i, (img, recon_img) in enumerate(zip(X, y_pred)):
        axs[2 * i].imshow(np.squeeze(np.asarray(img)))
        axs[2 * (i + 1)].imshow(np.squeeze(np.asarray(recon_img)))

    plt.savefig("./saved_models/test.png")
