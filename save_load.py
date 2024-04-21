import os
import yaml
import torch


def load_config(path: str):
    """
    Parse the config and return as dict
    Parameters
    ----------
    path: str
        Path to config yml

    Returns
    -------
    Nested dict containing config fields.
    """
    with open(path, "r") as cfg:
        try:
            ll = yaml.safe_load(cfg)
        except yaml.YAMLError as exc:
            print(exc)
    return ll


def save_checkpoint(save_path, model, optimizer, loss_dict, epoch_number):
    """SAVE_CHECKPOINT - save a model checkpoint as .pth

    Args:
        save_path (string): full path to .tar to be saved
        model (nn.Module): model to be saved
        optimizer (nn.Module): optimizer to be saved
        loss_dict (dict): contains loss history train and val, recon and kl
        epoch_number (int): epoch index to label file. Also saved in checkpoint dict
    """
    save_dir, _ = os.path.split(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = {"model": model.state_dict(),
                  "epsilon": model.epsilon,
                  "optimizer": optimizer.state_dict(),
                  "loss_dict": loss_dict,
                  "epoch": epoch_number
                  }
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
    model.epsilon = checkpoint["epsilon"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    loss_dict = checkpoint["loss_dict"]
    epoch = checkpoint["epoch"]

    print("Resuming from epoch: " + str(checkpoint["epoch"]))

    return model, optimizer, loss_dict, epoch


def create_output_directories(config):
    for d in ['model_save_dir', 'plot_save_dir']:
        if not os.path.exists(config['logging'][d]):
            os.makedirs(config['logging'][d])
