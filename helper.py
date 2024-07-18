from torchvision.utils import save_image
import os
from collections import namedtuple
import torch
from torch.nn import Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module
import numpy as np

def to_tensor(array):
    assert isinstance(array, np.ndarray)
    return torch.from_numpy(array).type(torch.FloatTensor).cuda()





def sample_images(logs_dir, source_img,syn_img,iter):
    """Saves generated samples"""
    save_path = os.path.join(logs_dir, 'samples')
    os.makedirs(save_path, exist_ok=True)

    save_image(
        source_img,
        os.path.join(save_path, f'{iter}_source.png'),
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )
    save_image(
        syn_img,
        os.path.join(save_path,f'{iter}_syn.png'),
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )

