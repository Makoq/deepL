import numpy as np
import torch
import torchvision.datasets as dest
import torchvision.transforms as transforms
import torchvision.utils as vutils

import const

def dateset_load():
    dataset = dest.ImageFolder(const.dateset_dir, transform=transforms.Compose([
        transforms.Resize(const.image_size),
        transforms.CenterCrop(const.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=const.batch_size,
                                         shuffle=True, num_workers=const.workers)

    return dataloader


    