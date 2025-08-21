from typing import Union, Tuple

import torch
from torch import NoneType

from FunctionEncoder import BaseDataset


class GaussianDonutDataset(BaseDataset):

    def __init__(self, radius=0.9, noise=0.1, device:str="auto"):
        super().__init__(input_size=(2,), # this distribution is not conditioned on anything, so we just want to predict the pdf for all two-d inputs
                         output_size=(1,),
                         data_type="pdf",
                         device=device,
                         n_examples=200,
                         n_queries=1_000,
                         )
        self.radius = radius
        self.noise = noise
        self.lows = torch.tensor([-1, -1], device=self.device)
        self.highs = torch.tensor([1, 1], device=self.device)
        self.positive_logit = 5
        self.negative_logit = -5
        self.volume = (self.highs - self.lows).prod()

    def __len__(self):
        return 1000

    def __getitem__(self, idx) -> Tuple[  torch.tensor,
                                torch.tensor, 
                                torch.tensor, 
                                torch.tensor,
                                dict]:
        # sample radiuses
        radii = torch.rand((1), device=self.device) * self.radius

        # generate example set
        angles = torch.rand((self. n_examples//2), device=self.device) * 2 * 3.14159
        example_xs = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=-1)
        example_xs += torch.randn_like(example_xs) * self.noise

        # generate point set
        angles = torch.rand((self. n_queries//2), device=self.device) * 2 * 3.14159
        xs = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=-1)
        xs += torch.randn_like(xs) * self.noise

        # concatenate points not from this distribution
        example_xs2 = self.sample_inputs(n_points=self. n_examples//2)
        xs2 = self.sample_inputs( n_points=self. n_queries//2)
        example_xs = torch.cat([example_xs, example_xs2], dim=0)
        xs = torch.cat([xs, xs2], dim=0)

        # give high logit to sampled points, and low logit to others
        example_ys = torch.zeros((self. n_examples, 1), device=self.device)
        example_ys[ :self. n_examples//2] = self.positive_logit
        example_ys[ self. n_examples//2:] = self.negative_logit
        ys = torch.zeros(( self. n_queries, 1), device=self.device)
        ys[ :self. n_queries//2] = self.positive_logit
        ys[ self. n_queries//2:] = self.negative_logit

        # move device
        info = {"radii": radii}
        example_xs = example_xs.to(self.device, dtype=self.dtype)
        example_ys = example_ys.to(self.device, dtype=self.dtype)
        xs = xs.to(self.device, dtype=self.dtype)
        ys = ys.to(self.device, dtype=self.dtype)


        return example_xs, example_ys, xs, ys, info

    def sample_inputs(self, n_points):
        inputs = torch.rand(n_points, 2, device=self.device) * (self.highs - self.lows) + self.lows
        return inputs