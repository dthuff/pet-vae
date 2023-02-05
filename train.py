import os
import torch
from plotting import plot_examples
scaler = torch.cuda.amp.GradScaler()


def train_loop(dataloader, model, loss_fn_kl, loss_fn_recon, optimizer, amp_on):
    """TRAIN_LOOP - Runs training for one epoch
    
    Args:
        dataloader (torch.DataLoader): dataloader for training set
        model (nn.Module): model object
        loss_fn_kl (nn.Module): Kullbeck-Liebler divergence loss
        loss_fn_recon (nn.Module): Reconstruction loss - e.g. L1, L2 (mse)
        optimizer (torch.Optimizer): model optimizer
        amp_on (Boolean): enable automatic mixed precision
    """

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        # Send the inputs X and labels y to the GPU
        X = X.cuda()
        y = y.cuda()

        # Compute prediction and loss
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_on):
            y_pred, z_mean, z_log_sigma = model(X)
            loss_kl = loss_fn_kl(z_mean, z_log_sigma)
            loss_recon = loss_fn_recon(y_pred, y)
            loss = loss_kl + loss_recon  # Consider weights here. IDK which loss is gonna dominate

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()
        optimizer.zero_grad()

        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Total loss: {loss:.2f}  [{current:>2d}/{size:>2d}]")
            print(f"    KL loss:{loss_kl.item():>20.2f}")
            print(f"    Recon loss:{loss_recon.item():>20.2f}")


def val_loop(dataloader, model, loss_fn_kl, loss_fn_recon):
    """VAL_LOOP - Runs validation for one epoch

    Args:
        dataloader (torch.DataLoader): dataloader for validation set
        model (nn.Module): model object
        loss_fn_kl (nn.Module): Kullbeck-Liebler divergence loss
        loss_fn_recon (nn.Module): Reconstruction loss - e.g. L1, L2 (mse)
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_kl, loss_recon = 0, 0
    plotted_this_epoch = False

    with torch.no_grad():

        for X, y in dataloader:
            # Send the inputs X and labels y to the GPU
            X = X.cuda()
            y = y.cuda()

            # Compute prediction and loss
            y_pred, z_mean, z_log_sigma = model(X)
            loss_kl += loss_fn_kl(z_mean, z_log_sigma).item()
            loss_recon += loss_fn_recon(y_pred, y).item()

            # TODO - plot some X and y_pred montages
            if not plotted_this_epoch:
                plot_examples(X.cpu(), y_pred.cpu())
                plotted_this_epoch = True
            # TODO - plots N(0,1) against N(z_mean, z_log_sigma)
    loss_kl /= num_batches
    loss_recon /= num_batches

    return loss_kl, loss_recon


# TODO test_loop
def test_loop(dataloader, model, loss_fn_kl, loss_fn_recon):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
