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

def record(recorded: list, module, _, output):
    recorded.append(output)

def get_input_hooks():

    hidden_to_be_consumed = None

    def set_hidden(hidden: Tensor | None):
        nonlocal hidden_to_be_consumed
        hidden_to_be_consumed = hidden

    def hook(module, inp, output):
        input_is_tuple = isinstance(inp, tuple)

        if input_is_tuple:
            inp, *rest = inp

        if exists(hidden_to_be_consumed):
            inp = inp + hidden_to_be_consumed

        return inp

    return set_hidden, hook

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

        hooks = []

        # add hook for output module on model
        # for extracting the hidden state to be passed back into the model

        output_hook = output_module_or_path.register_forward_hook(partial(record, self.hiddens))
        hooks.append(output_hook)

        # wire up the input module so it receives the hidden state from previous timestep from output module hook above

        self.set_hidden_for_input, input_forward_hook = get_input_hooks()

        input_hook = input_module_or_path.register_forward_hook(input_forward_hook)
        hooks.append(input_hook)

        self.hooks = hooks

    def unregister_hooks(self):
        for hook in self.hooks:
            hook.remove()

        self.hooks = None

    def forward(
        self,
        *args,
        hiddens_for_rnn: Tensor | None = None,
        **kwargs
    ):
        assert exists(self.hooks), 'no hooks registered'

        self.set_hidden_for_input(hiddens_for_rnn)

        # run inputs through the model as usual
        # hooks should be triggered for input to incorporate the hidden being passed in, and then also record the hidden state output

        out = self.model(*args, **kwargs)

        # get the next hidden state

        hidden, = self.hiddens

        self.hiddens.clear()
        self.set_hidden_for_input(None)

        return out, hidden

# main class

class GameNGen(Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x
