"""get device lib"""

import torch

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
