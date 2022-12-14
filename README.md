# EDEN compression

[![PyPI](https://img.shields.io/pypi/v/cmprs-eden.svg)](https://pypi.org/project/cmprs-eden/)
[![Changelog](https://img.shields.io/github/v/release/cmprs/eden?include_prereleases&label=changelog)](https://github.com/cmprs/eden/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/cmprs/eden/blob/main/LICENSE)

A PyTorch implementation of 'EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated
Learning', presented at [ICML 2022](https://proceedings.mlr.press/v162/vargaftik22a.html).

## Installation

Install this library using `pip`:

    pip install cmprs-eden

## Usage

```python
from cmprs_eden import eden
import torch

from dataclasses import astuple

compression = eden(bits=1)

x = torch.randn(2 ** 12)

compressed_x, context, bitrate = astuple(compression.forward(x))
reconstructed_x = compression.backward(compressed_x, context)
```

## Development

To contribute to this library, first checkout the code. Then create a new virtual environment:

    cd <project-checkout-folder>
    python -m venv venv
    source venv/bin/activate

Now install the development dependencies:

    pip install -e ".[test,preprocess]"

### To run the tests:

    pytest

## Citation

If you find this useful, please cite us:

```bibtex
@InProceedings{pmlr-v162-vargaftik22a,
  title = 	 {{EDEN}: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning},
  author =       {Vargaftik, Shay and Basat, Ran Ben and Portnoy, Amit and Mendelson, Gal and Itzhak, Yaniv Ben and Mitzenmacher, Michael},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {21984--22014},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/vargaftik22a/vargaftik22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/vargaftik22a.html},
  abstract = 	 {Distributed Mean Estimation (DME) is a central building block in federated learning, where clients send local gradients to a parameter server for averaging and updating the model. Due to communication constraints, clients often use lossy compression techniques to compress the gradients, resulting in estimation inaccuracies. DME is more challenging when clients have diverse network conditions, such as constrained communication budgets and packet losses. In such settings, DME techniques often incur a significant increase in the estimation error leading to degraded learning performance. In this work, we propose a robust DME technique named EDEN that naturally handles heterogeneous communication budgets and packet losses. We derive appealing theoretical guarantees for EDEN and evaluate it empirically. Our results demonstrate that EDEN consistently improves over state-of-the-art DME techniques.}
}
```
