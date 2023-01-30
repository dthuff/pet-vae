from dataloader import DicomDataset
from torch.utils.data import DataLoader
#import matplotlib.pyplot as plt

data_path = '/home/daniel/Projects/pet-vae/data/acrin-nsclc-fdg-cleaned/'
my_dataset = DicomDataset(data_path)

my_dataloader = DataLoader(my_dataset, batch_size=16, shuffle=True)

for i in range(10):
    train_features, train_labels = next(iter(my_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

a = 5