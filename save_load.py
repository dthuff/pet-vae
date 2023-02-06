import os
import torch


def save_checkpoint(save_path, model, optimizer, loss_kl, loss_recon, epoch_number):
    """SAVE_CHECKPOINT - save a model checkpoint as .tar

    Args:
        save_path (string): full path to .tar to be saved
        model (nn.Module): model to be saved
        optimizer (nn.Module): optimizer to be saved
        loss_kl (_type_): KL loss at this point - you pick val or train loss
        loss_recon (_type_): Reconstruction loss at this point - you pick val or train loss
        epoch_number (int): epoch index to label file. Also saved in checkpoint dict
    """
    # Create a save directory if it does not yet exist
    save_dir, _ = os.path.split(save_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # Write the model to a dictionary
    checkpoint = {"model": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "loss_KL": loss_kl,
                  "loss_recon": loss_recon,
                  "epoch": epoch_number}

    # Save
    torch.save(checkpoint, save_path)


def load_from_checkpoint(checkpoint_path, model, optimizer):
    """LOAD_FROM_CHECKPOINT - load the state dicts for an initialized model and optimizer
        The model and optimizer must be initialized before calling this
        
    Args:
        checkpoint_path (string): path to model checkpoint .tar
        model (nn.Module): An initialized instance of the model object
        optimizer (nn.Module): An initialized instance of the optimizer

    Returns:
        model (nn.Module): The model object with state_dict loaded from checkpoint
        optimizer (nn.Module): The optimized object with state_dict loaded from checkpoint
    """
    # Load the checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location={"cpu": "cuda:0"})

    # Apply the loaded state dicts to model and optimizer
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    print("Resuming from epoch: " + str(checkpoint["epoch"]))

    return model, optimizer