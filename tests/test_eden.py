import math
from dataclasses import astuple

import torch

from cmprs_core import config
from cmprs_eden import eden


def normal_generator(d):
    dist = torch.distributions.Normal(0, 1)
    return lambda n_clients: dist.sample([n_clients, d]).to(config.device)


sample_vec_fn = normal_generator(2 ** 12)


def estimate_vNMSE(x, repeats, compression_scheme):
    total_SSE = 0
    for r in range(repeats):
        est_x = compression_scheme.roundtrip(x).rx
        total_SSE += torch.sum((est_x - x) ** 2)
    return total_SSE / repeats / torch.sum(x ** 2)


def test_eden_1b():
    x = sample_vec_fn(1)
    assert math.isclose(estimate_vNMSE(x, 10, eden(bits=1)).item(), 0.57079, abs_tol=0.005)

def test_readme():
    compression = eden(bits=1)

    x = torch.randn(2 ** 12)

    compressed_x, context, bitrate = astuple(compression.forward(x))
    reconstructed_x = compression.backward(compressed_x, context)
