import torch
from torchvision import transforms
import numpy as np
import os, shutil
import time
import random
import pandas as pd 
from torch.utils.data import Dataset
from pydicom import dcmread
from glob import glob

class DicomDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None) -> None:
        self.img_dir = img_dir
        self.img_list = glob(self.img_dir + '*/PET/*.dcm')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return(len(self.img_list))
        #return len(os.listdir(self.img_dir))

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_list[index])
        image = dcmread(img_path).pixel_array
        image = image.astype('float32')
        label = image
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label