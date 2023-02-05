import torch
from dataloader import DicomDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ConvertImageDtype
from model import VAE
from loss import KLDivergence, L2Loss
from train import train_loop, test_loop
#import matplotlib.pyplot as plt

# Hyperparameters:
batch_size = 16
learning_rate = 0.001
max_epochs = 100
weight_decay = 5e-7
device = 'cuda'
data_path = '/home/daniel/Projects/pet-vae/data/ACRIN-NSCLC-FDG-PET-cleaned/'

transform_composition = Compose([
                            ToTensor(),
                            Resize(128),
                            ConvertImageDtype(torch.float)
                        ])

train_dataset = DicomDataset(data_path, 
                            transform=transform_composition,
                            target_transform=transform_composition)

train_dataloader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True)

'''
for batch, (X, y) in enumerate(train_dataloader):
    print(batch)
'''

model = VAE()
model.to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# train_features, train_labels = next(iter(train_dataloader))

for t in range(max_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloader=train_dataloader, 
                model=model, 
                loss_fn_KL=KLDivergence(), 
                loss_fn_recon=L2Loss(), 
                optimizer=optimizer,
                device=device)
    # test_loop(test_dataloader, model, loss_fn)
print("Done!")