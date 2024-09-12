from __future__ import annotations
from functools import partial

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

def record(recorded: list, module, input, output):
    recorded.append(output)

class RNNify(Module):
    def __init__(
        self,
        model: Module,
        output_module_or_path: str | Module,
        input_module_or_path: str | Module
    ):
        super().__init__()
        self.model = model
        self.hiddens = []

        name_to_module = {name: module for name, module in model.named_modules()}

        if isinstance(input_module_or_path, str):
            assert input_module_or_path in name_to_module, f'{input_module_or_path} not found'
            input_module_or_path = name_to_module[input_module_or_path]

        if isinstance(output_module_or_path, str):
            assert output_module_or_path in name_to_module, f'{output_module_or_path} not found'
            output_module_or_path = name_to_module[output_module_or_path]

        output_module_or_path.register_forward_hook(partial(record, self.hiddens))

    def forward(
        self,
        *args,
        **kwargs
    ):
        out = self.model(*args, **kwargs)

        hidden, = self.hiddens
        self.hiddens.clear()

        return out, hidden

# main class

class GameNGen(Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x
