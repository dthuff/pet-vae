import torch
import numpy as np
from skimage.metrics import structural_similarity


def calculate_psnr(original, reconstruction):
    """
    Compute Peak signal-to-noise between the original image and the reconstructed image
    Args:
        original: tensor
            input image. Shape is batch, channel, h, w
        reconstruction: tensor
            VAE output reconstruction. Shape is batch, channel, h, w

    Returns:
        psnr : float
            The PSNR in decibels
    """

    mse = torch.mean((original - reconstruction) ** 2).cpu().detach().numpy()
    psnr = 20 * np.log10(1 / np.sqrt(mse))
    return psnr


def calculate_ssim(original, reconstruction):
    """
    Compute the structural similarity between the original image and the reconstructed image.
    Args:
        original: tensor
            input image. Shape is batch, channel, h, w
        reconstruction: tensor
            VAE output reconstruction. Shape is batch, channel, h, w

    Returns:
        ssim : float
            The SSIM - between 0 and 1, higher is better.
    """
    original = original.cpu().detach().squeeze().numpy()
    reconstruction = reconstruction.cpu().detach().squeeze().numpy()

    ssim = np.zeros((original.shape[0],))
    ssim_norm = np.zeros((original.shape[0],))

    for sl in range(original.shape[0]):
        original_slice = np.squeeze(original[sl,:,:])
        reconstruction_slice = np.squeeze(reconstruction[sl,:,:])

        original_slice_norm = (original_slice - np.mean(original_slice)) / np.std(original_slice)
        reconstruction_slice_norm = (reconstruction_slice - np.mean(reconstruction_slice)) / np.std(reconstruction_slice)

        ssim[sl] = structural_similarity(original_slice, reconstruction_slice)
        ssim_norm[sl] = structural_similarity(original_slice_norm, reconstruction_slice_norm)

    return ssim, ssim_norm
