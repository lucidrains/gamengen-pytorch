import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# class

class GameNGen(Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x
