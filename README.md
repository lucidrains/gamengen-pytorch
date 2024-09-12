## GameNGen - Pytorch (wip)

Implementation of a framework for <a href="https://gamengen.github.io/">Gamengen</a> in Pytorch

## Usage

```python
import torch

# mock model using x-transformer

from x_transformers import Encoder
model = Encoder(dim = 512, depth = 6)

# get RNNify module wrapper

from gamengen_pytorch import RNNify

# wrap the model, and pass in the module names where hidden state

rnn = RNNify(
    model,
    input_module_or_path = 'layers.0.2',
    output_module_or_path = 'layers.11.2',
)

x = torch.randn(1, 1024, 512)

out1, hiddens1 = rnn(x)
out2, hiddens2 = rnn(x, hiddens_for_rnn = hiddens1.detach())
out3, hiddens3 = rnn(x, hiddens_for_rnn = hiddens2.detach())
```

## Citations

```bibtex
@inproceedings{Valevski2024DiffusionMA,
    title   = {Diffusion Models Are Real-Time Game Engines},
    author  = {Dani Valevski and Yaniv Leviathan and Moab Arar and Shlomi Fruchter},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:271962839}
}
```

```bibtex
@article{Ding2024DiffusionWM,
    title     = {Diffusion World Model},
    author    = {Zihan Ding and Amy Zhang and Yuandong Tian and Qinqing Zheng},
    journal   = {ArXiv},
    year      = {2024},
    volume    = {abs/2402.03570},
    url       = {https://api.semanticscholar.org/CorpusID:267499902}
}
```
