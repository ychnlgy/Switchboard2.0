import torch

DEVICE = ["cpu", "cuda"][torch.cuda.is_available()]
