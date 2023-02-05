import torch

def train_loop(dataloader, model, loss_fn_KL, loss_fn_recon, optimizer, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        # Send the inputs X and labels y to the GPU
        X = X.cuda()
        y = y.cuda()

        # Compute prediction and loss
        y_pred, z_mean, z_log_sigma = model(X)
        loss_KL = loss_fn_KL(z_mean, z_log_sigma)
        loss_recon = loss_fn_recon(y_pred, y)
        loss = loss_KL + loss_recon # Consider weights here. IDK which loss is gonna dominate

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f"KL loss: {loss_KL.item():>4f}")
            print(f"Recon loss: {loss_recon.item():>4f}")
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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