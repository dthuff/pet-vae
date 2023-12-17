import torch
import os

from performance_metrics import calculate_psnr, calculate_ssim
from plotting import plot_examples

scaler = torch.cuda.amp.GradScaler()


def train_loop(dataloader, model, loss_fn_kl, loss_fn_recon, beta, optimizer, amp_on):
    """TRAIN_LOOP - Runs training for one epoch

    VAE loss L is:

        L = L_recon(y, y_pred) + beta * L_KL(mu, sigma)

    Args:
        dataloader (torch.DataLoader): dataloader for training set
        model (nn.Module): model object
        loss_fn_kl (nn.Module): Kullbeck-Liebler (KL) divergence loss
        loss_fn_recon (nn.Module): Reconstruction loss - e.g. L1, L2 (mse)
        beta: (float): Weight for KL loss term
        optimizer (torch.Optimizer): model optimizer
        amp_on (Boolean): enable automatic mixed precision

    Returns:
        mean_loss_kl
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    total_loss_kl = 0
    total_loss_recon = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.cuda()
        y = y.cuda()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_on):
            y_pred, z_mean, z_log_sigma = model(X)
            batch_loss_kl = beta * loss_fn_kl(z_mean, z_log_sigma)
            batch_loss_recon = loss_fn_recon(y_pred, y)
            batch_loss = batch_loss_recon + batch_loss_kl

        # Backpropagation with GradScaler for optional automatic mixed precision
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss_kl += batch_loss_kl.item()
        total_loss_recon += batch_loss_recon.item()

        if batch % round(0.2 * num_batches) == 0:
            loss, current = batch_loss.item(), batch * len(X)
            print(f"Total loss: {loss:.2f}  [{current:>2d}/{size:>2d}]")
            print(f"    KL loss:   {batch_loss_kl.item():>20.2f}")
            print(f"    Recon loss:{batch_loss_recon.item():>20.2f}")

    return total_loss_kl / num_batches, total_loss_recon / num_batches


def val_loop(dataloader, model, loss_fn_kl, loss_fn_recon, beta, epoch_number, plot_every_n_epochs, plot_save_dir):
    """VAL_LOOP - Runs validation for one epoch

    Args:
        dataloader (torch.DataLoader): dataloader for validation set
        model (nn.Module): model object
        loss_fn_kl (nn.Module): Kullbeck-Liebler divergence loss
        loss_fn_recon (nn.Module): Reconstruction loss - e.g. L1, L2 (mse)
        beta: (float): Weight for KL loss term
        epoch_number (int): epoch counter for saving plots
        plot_every_n_epochs (int): frequency to plot example images
        plot_save_dir (str): folder to save example images to
    """
    # Stop training during validation
    model.eval()

    num_batches = len(dataloader)
    loss_kl, loss_recon = 0, 0
    plotted_this_epoch = False

    with torch.no_grad():
        for X, y in dataloader:
            X = X.cuda()
            y = y.cuda()

            # Compute prediction and loss
            y_pred, z_mean, z_log_sigma = model(X)
            loss_kl += beta * loss_fn_kl(z_mean, z_log_sigma).item()
            loss_recon += loss_fn_recon(y_pred, y).item()

            # Plot a montage of X and y_pred comparisons for the first batch
            if epoch_number % plot_every_n_epochs == 0 and not plotted_this_epoch:
                plot_examples(X=X.cpu(),
                              y_pred=y_pred.cpu(),
                              plot_path=os.path.join(plot_save_dir, f"val_epoch_{epoch_number}.png"))
                plotted_this_epoch = True

            # TODO - plots N(0,1) against N(z_mean, z_log_sigma)
    loss_kl /= num_batches
    loss_recon /= num_batches

    return loss_kl, loss_recon


def test_loop(dataloader, model, loss_fn_kl, loss_fn_recon, plot_save_dir):
    """

    Args:
        dataloader:
        model:
        loss_fn_kl:
        loss_fn_recon:
        plot_save_dir:

    Returns:
        loss_kl:
        loss_recon:
        perf_metric : dict
            dictionary containing psnr and ssim per batch
    """
    # Do not train model at test time
    model.eval()

    num_batches = len(dataloader)
    loss_kl, loss_recon = 0, 0
    psnr = []  # A list of per-batch PSNR values
    ssim = []  # A list of per-image SSIM values

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.cuda()
            y = y.cuda()

            y_pred, z_mean, z_log_sigma = model(X)
            loss_kl += loss_fn_kl(z_mean, z_log_sigma).item()
            loss_recon += loss_fn_recon(y_pred, y).item()

            # Plot a montage of X and y_pred comparisons
            plot_examples(X=X.cpu(),
                          y_pred=y_pred.cpu(),
                          plot_path=os.path.join(plot_save_dir, f"test_examples_batch_{batch}.png"))

            psnr.append(calculate_psnr(y, y_pred))
            ssim.append(calculate_ssim(y, y_pred))

    loss_kl /= num_batches
    loss_recon /= num_batches

    perf_metrics = dict()
    perf_metrics["ssim"] = ssim
    perf_metrics["psnr"] = psnr

    return loss_kl, loss_recon, perf_metrics
