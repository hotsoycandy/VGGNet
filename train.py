"""VGG Train"""

from matplotlib import pyplot as plt
import numpy as np
## torches
import torch
from torch import nn
from torchvision import transforms as torchTrans
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchsummary import summary as torchsummary

def get_device ():
    """
    get torch processing device
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    return device

device = get_device()
print(device)

def train ():
    pass

train()
