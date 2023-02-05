import os
import torch
from torch.cuda.amp import GradScaler

scaler = GradScaler()

def train_loop(dataloader, model, loss_fn_KL, loss_fn_recon, optimizer, amp_on):
    """TRAIN_LOOP - Runs training for one epoch
    
    Args:
        dataloader (torch.DataLoader): dataloader for training set
        model (nn.Module): model object
        loss_fn_KL (nn.Module): Kullbeck-Liebler divergence loss
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
            loss_KL = loss_fn_KL(z_mean, z_log_sigma)
            loss_recon = loss_fn_recon(y_pred, y)
            loss = loss_KL + loss_recon # Consider weights here. IDK which loss is gonna dominate

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            print(f"KL loss: {loss_KL.item():>4f}")
            print(f"Recon loss: {loss_recon.item():>4f}")
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def val_loop(dataloader, model, loss_fn_KL, loss_fn_recon):
    """VAL_LOOP - Runs validation

    Args:
        dataloader (torch.DataLoader): dataloader for validation set
        model (nn.Module): model object
        loss_fn_KL (nn.Module): Kullbeck-Liebler divergence loss
        loss_fn_recon (nn.Module): Reconstruction loss - e.g. L1, L2 (mse)
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss_KL, loss_recon = 0, 0

    with torch.no_grad():
        for X, y in dataloader:

            # Send the inputs X and labels y to the GPU
            X = X.cuda()
            y = y.cuda()

            # Compute prediction and loss
            y_pred, z_mean, z_log_sigma = model(X)
            loss_KL += loss_fn_KL(z_mean, z_log_sigma).item()
            loss_recon += loss_fn_recon(y_pred, y).item()

            # TODO - plot some X and y_pred montages

            # TODO - plots N(0,1) against N(z_mean, z_log_sigma)
    loss_KL /= num_batches
    loss_recon /= num_batches
    print(f"Val loss: \n   KL loss: {(loss_KL):>0.1f}\n   Recon loss: {loss_recon:>8f} \n")

# TODO test_loop
def test_loop(dataloader, model, loss_fn):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")