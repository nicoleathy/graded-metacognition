import numpy as np
import torch
import torch.nn as nn


def apply_evolution(
    model: nn.Module,
    seeds: list[int] | int | np.ndarray,
    absolute_scale: float,
    relative_scales: list[float] = [1.0],
    reverse: bool = False,
):
    if isinstance(seeds, int) or seeds.ndim == 0:
        seeds = [seeds]
    gen = torch.Generator(device=model.device)
    for idx, seed in enumerate(seeds):
        gen.manual_seed(int(seed))

        for param in model.parameters():
            update = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
            update.mul_(relative_scales[idx] * absolute_scale / len(seeds))

            if reverse:
                param.data.sub_(update)
            else:
                param.data.add_(update)
