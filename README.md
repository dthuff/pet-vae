# pet-vae
2D variational autoencoder for reconstructing PET images implemented in pytorch

Trained using data from ACRIN-NSCLC-FDG-PET hosted by The Cancer Imaging Archive. Data available from: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=39879162

Converges in ~3-5 hours training on a NVIDIA 1080 (8 GB).

Figure 1: randomly selected example axial PET slices from the validation set after 540 epochs. Odd columns are model input, even columns are model reconstructions.
![alt text](https://github.com/dthuff/pet-vae/blob/master/saved_models/validation_images/epoch_540.png?raw=true)


Figure 2: Loss curve for the VAE. Total loss was the sum of Kullback-Liebler (KL) and reconstruction L2 (RECON) losses.
![alt text](https://github.com/dthuff/pet-vae/blob/master/saved_models/loss.png?raw=true)
