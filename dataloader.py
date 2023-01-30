import torch
import numpy as np
import torchio as tio
import os, shutil
import time
import random
import pandas as pd 
from torch.utils.data import Dataset
from glob import glob

class DicomDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None) -> None:
        self.img_dir = img_dir
        self.img_list = glob(self.img_dir + '*/PET/*.dcm')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        image = tio.ScalarImage(img_path).data
        label = image
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label