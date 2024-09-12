import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class RNNify(Module):
    def __init__(
        self,
        model: Module,
        output_hidden_module_path: str,
        input_hidden-module_path: str
    ):
        super().__init__()

        self.model = model

    def forward(
        self,
        *args,
        **kwargs
    ):
        out = self.model(*args, **kwargs)
        return out

# main class

class GameNGen(Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x
