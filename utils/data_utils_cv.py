# -*- coding: utf-8 -*-
"""
@author: Eva Pachetti
"""

import logging
import torch # type: ignore
from torchvision import transforms # type: ignore
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler # type: ignore
from create_dataset import ProstateDataset, ToTensorDataset 
import numpy as np # type: ignore
import os
from tools import normalize

logger = logging.getLogger(__name__)


def get_loader(args,cv):
 
    csv_file_train = os.path.join(args.csv_path, f"training_cv{cv+1}.csv")
    csv_file_val = os.path.join(args.csv_path, f"validation_cv{cv+1}.csv")

    trainset = list(ProstateDataset(csv_file_train))
    validset = list(ProstateDataset(csv_file_val))

    volumes_train = [item[0] for item in trainset]
    mean = np.mean(volumes_train)

    # AUGMENTATION
    aug_suffixes = ['rotation', 'vertical_flip', 'horizontal_flip']
    trainsets_aug = [ProstateDataset(csv_file_train, aug_folder=aug) for aug in aug_suffixes]

    # Only for PROSTATEX
    hg_positions = [idx for idx, item in enumerate(trainset) if item[1].item() == 1]
    positions = hg_positions[::3]

    for trainset_aug in trainsets_aug:
        trainset_aug = list(trainset_aug)
        for position in positions:
            trainset.append(trainset_aug[position])

  
    trainset = normalize(trainset, mean)
    validset = normalize(validset, mean)

    trainset = ToTensorDataset(trainset)
    validset = ToTensorDataset(validset)

    train_sampler = RandomSampler(trainset) 
    valid_sampler = SequentialSampler(validset)
    train_loader = DataLoader(trainset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(validset, sampler=valid_sampler, batch_size=args.eval_batch_size, num_workers=0, pin_memory=True) if validset else None

    return train_loader, valid_loader
