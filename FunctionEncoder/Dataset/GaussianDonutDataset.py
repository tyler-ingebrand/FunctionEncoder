from typing import Union, Tuple

import torch
from torch import NoneType

from FunctionEncoder import BaseDataset


class GaussianDonutDataset(BaseDataset):

    def __init__(self, radius=0.9, noise=0.1):
        super().__init__(input_size=(0,), # this distribution is not conditioned on anything, so we just want to predict the pdf for all two-d inputs
                         output_size=(2,),
                         total_n_functions=float("inf"),
                         total_n_samples_per_function=float("inf"),
                         data_type="stochastic",
                         n_functions_per_sample=10,
                         n_examples_per_sample=100,
                         n_points_per_sample=10_000,
                         )
        self.radius = radius
        self.noise = noise
        self.lows = torch.tensor([-1, -1])
        self.highs = torch.tensor([1, 1])
        self.volume = (self.highs - self.lows).prod()


    def sample(self, device:Union[str, torch.device]) -> Tuple[ Union[torch.tensor, NoneType],
                                                                Union[torch.tensor, NoneType],
                                                                Union[torch.tensor, NoneType],
                                                                Union[torch.tensor, NoneType],
                                                                dict]:
        # sample radiuses
        radii = torch.rand((self.n_functions_per_sample, 1), device=device) * self.radius

        # generate example set
        angles = torch.rand((self.n_functions_per_sample, self.n_examples_per_sample), device=device) * 2 * 3.14159
        example_ys = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=-1)
        example_ys += torch.randn_like(example_ys) * self.noise

        # generate point set
        angles = torch.rand((self.n_functions_per_sample, self.n_points_per_sample), device=device) * 2 * 3.14159
        ys = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=-1)
        ys += torch.randn_like(ys) * self.noise

        # We have no conditional variables, so examples are none in this case
        examples_xs = None
        xs = None

        # move device
        examples_ys = example_ys.to(device)
        ys = ys.to(device)
        info = {"radii": radii}
        return examples_xs, examples_ys, xs, ys, info

    def sample_inputs(self, n_functions, n_points, device):
        inputs = torch.rand(n_functions, n_points, 2) * (self.highs - self.lows) + self.lows
        return inputs.to(device)