import torch


def get_torch_device(extra=False):
    if extra:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print("Torch device:", device)
    return device
