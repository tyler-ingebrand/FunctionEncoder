from typing import Union, Tuple

import torch
from torch import NoneType

from FunctionEncoder import BaseDataset


class GaussianDonutDataset(BaseDataset):

    def __init__(self, radius=0.9, noise=0.1):
        super().__init__(input_size=(2,), # this distribution is not conditioned on anything, so we just want to predict the pdf for all two-d inputs
                         output_size=(1,),
                         total_n_functions=float("inf"),
                         total_n_samples_per_function=float("inf"),
                         data_type="stochastic",
                         n_functions_per_sample=10,
                         n_examples_per_sample=200,
                         n_points_per_sample=1_000,
                         )
        self.radius = radius
        self.noise = noise
        self.lows = torch.tensor([-1, -1])
        self.highs = torch.tensor([1, 1])
        self.positive_logit = 5
        self.negative_logit = -5
        self.volume = (self.highs - self.lows).prod()


    def sample(self, device:Union[str, torch.device]) -> Tuple[ torch.tensor, 
                                                                torch.tensor, 
                                                                torch.tensor, 
                                                                torch.tensor,
                                                                dict]:
        # sample radiuses
        radii = torch.rand((self.n_functions_per_sample, 1), device=device) * self.radius

        # generate example set
        angles = torch.rand((self.n_functions_per_sample, self.n_examples_per_sample//2), device=device) * 2 * 3.14159
        example_xs = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=-1)
        example_xs += torch.randn_like(example_xs) * self.noise

        # generate point set
        angles = torch.rand((self.n_functions_per_sample, self.n_points_per_sample//2), device=device) * 2 * 3.14159
        xs = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=-1)
        xs += torch.randn_like(xs) * self.noise

        # concatenate points not from this distribution
        example_xs2 = self.sample_inputs(n_functions=self.n_functions_per_sample, n_points=self.n_examples_per_sample//2, device=device)
        xs2 = self.sample_inputs(n_functions=self.n_functions_per_sample, n_points=self.n_points_per_sample//2, device=device)
        example_xs = torch.cat([example_xs, example_xs2], dim=1)
        xs = torch.cat([xs, xs2], dim=1)

        # give high logit to sampled points, and low logit to others
        example_ys = torch.zeros((self.n_functions_per_sample, self.n_examples_per_sample, 1), device=device)
        example_ys[:, :self.n_examples_per_sample//2] = self.positive_logit
        example_ys[:, self.n_examples_per_sample//2:] = self.negative_logit
        ys = torch.zeros((self.n_functions_per_sample, self.n_points_per_sample, 1), device=device)
        ys[:, :self.n_points_per_sample//2] = self.positive_logit
        ys[:, self.n_points_per_sample//2:] = self.negative_logit

        # move device
        example_xs = example_xs.to(device)
        example_ys = example_ys.to(device)
        xs = xs.to(device)
        ys = ys.to(device)
        info = {"radii": radii}
        return example_xs, example_ys, xs, ys, info

    def sample_inputs(self, n_functions, n_points, device):
        inputs = torch.rand(n_functions, n_points, 2) * (self.highs - self.lows) + self.lows
        return inputs.to(device)