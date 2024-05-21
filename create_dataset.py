# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""
    

import torch # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import os
from torch.utils.data import Dataset # type: ignore

class ProstateDataset(Dataset):
    def __init__(self, input, aug_folder="original", size=128, ood=False, mean=0, var=1, bootstrap=False):
        
        if bootstrap:
            self.info = input
        else:
            self.info = pd.read_csv(input)
        self.aug_folder = aug_folder
        self.dir_path = os.path.join(os.getcwd(),"dataset")
        self.size = size
        self.ood = ood
        self.mean = mean
        self.var = var

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        folder = self.info.iloc[idx, 0]
        volume_path = os.path.join(self.dir_path, folder, self.aug_folder)
        slices = os.listdir(volume_path)
        dtype = slices[0].split('.')[1]
        
        z = self.info.iloc[idx, 4]
        real_z = z if folder == 'ProstateX-0179' else len(slices) - z + 1
        slice_filename = f'1-{"0" if real_z <= 9 else ""}{real_z}.{dtype}'
        slice_idx = slices.index(slice_filename)
        five_slices = slices[max(0, slice_idx-2):slice_idx+3]
        
        volume = np.zeros((self.size, self.size, 5, 1))  # H, W, Z, C
        
        for k, s in enumerate(five_slices):
            image_path = os.path.join(volume_path, s)
            if dtype == 'png':
                v = np.array(Image.open(image_path))
            elif dtype == 'npy':
                v = np.load(image_path)

            if self.ood:
                row, col = v.shape
                gauss = np.random.normal(self.mean, self.var, (row, col))
                v = v + gauss

            volume[:, :, k, 0] = v

        label_str = str(self.info.iloc[idx, 1])
        label = np.array(0 if label_str == 'LG' else 1, dtype=int)
        label = torch.tensor(label)
        
        patient = self.info.iloc[idx, 0]
        zone = self.info.iloc[idx, 2]
        
        return volume, label, patient, zone
    


class ToTensorDataset(torch.utils.data.Subset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        volume, label = self.dataset[idx][:2]  # Slicing for volume and label
        volume = torch.from_numpy(volume).float().permute(3, 0, 1, 2)  # Convert to tensor and permute dimensions

        return volume, label

    def __len__(self):
        return len(self.dataset)




        