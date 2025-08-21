from typing import Union, Tuple

import torch
from torch import NoneType

from FunctionEncoder import BaseDataset


class GaussianDataset(BaseDataset):

    def __init__(self, device:str="auto"):
        super().__init__(input_size=(2,), # this distribution is not conditioned on anything, so we just want to predict the pdf for all two-d inputs
                         output_size=(1,),
                         data_type="pdf",
                         device=device,
                         n_examples=100,
                         n_queries=500,
                         )
        self.lows = torch.tensor([-1, -1], device=self.device)
        self.highs = torch.tensor([1, 1], device=self.device)
        self.positive_logit = 5
        self.negative_logit = -5
        self.volume = (self.highs - self.lows).prod()
        self.noise_min = 0.05
        self.noise_max = 0.35

    def __len__(self):
        return 1000

    def __getitem__(self, idx) -> Tuple[  torch.tensor,
                                torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                dict]:
        # sample radiuses
        std_devs = torch.rand( 1, device=self.device) * (self.noise_max - self.noise_min) + self.noise_min

        # generate example set
        example_xs = torch.randn(( self.n_examples//2, 2), device=self.device) * std_devs.reshape(1, 1)

        # generate point set
        query_xs = torch.randn(( self.n_queries//2, 2), device=self.device) * std_devs.reshape(1, 1)

        # concatenate points not from this distribution
        example_xs2 = self.sample_inputs(n_points=self.n_examples//2, points=example_xs)
        query_xs2 = self.sample_inputs(n_points=self.n_queries//2, points=query_xs)
        example_xs = torch.cat([example_xs, example_xs2], dim=0)
        query_xs = torch.cat([query_xs, query_xs2], dim=0)

        # give high logit to sampled points, and low logit to others
        example_ys = torch.zeros(( self.n_examples, 1), device=self.device)
        example_ys[ :self.n_examples//2] = self.positive_logit
        example_ys[ self.n_examples//2:] = self.negative_logit
        query_ys = torch.zeros(( self.n_queries, 1), device=self.device)
        query_ys[ :self.n_queries//2] = self.positive_logit
        query_ys[ self.n_queries//2:] = self.negative_logit

        info = {"std_devs": std_devs}
        return example_xs, example_ys, query_xs, query_ys, info

    def sample_inputs(self, n_points, points):
        points = torch.rand(n_points, 2, device=self.device) * (self.highs - self.lows) + self.lows
        return points