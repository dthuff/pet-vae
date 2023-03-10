import torch
from torchviz import make_dot

from model import VAE
device = "cuda:0"

model = VAE()
model.to(device=device)

# Dummy tensor for batch size 16, 1 channel, image size 128 x 128.
x = torch.randn(16, 1, 128, 128)
x = x.to(device=device)
y = model(x)

make_dot(y, params=dict(model.named_parameters())).render("vae.png")