from typing import Union, Tuple

import torch
from torch import NoneType

from FunctionEncoder import BaseDataset


class GaussianDataset(BaseDataset):

    def __init__(self):
        super().__init__(input_size=(2,), # this distribution is not conditioned on anything, so we just want to predict the pdf for all two-d inputs
                         output_size=(1,),
                         total_n_functions=float("inf"),
                         total_n_samples_per_function=float("inf"),
                         data_type="stochastic",
                         n_functions_per_sample=10,
                         n_examples_per_sample=1000,
                         n_points_per_sample=1_000,
                         )
        self.lows = torch.tensor([-1, -1])
        self.highs = torch.tensor([1, 1])
        self.positive_logit = 5
        self.negative_logit = -5
        self.volume = (self.highs - self.lows).prod()
        self.noise_min = 0.05
        self.noise_max = 0.35


    def sample(self, device:Union[str, torch.device]) -> Tuple[ Union[torch.tensor, type(None)],
                                                                Union[torch.tensor, type(None)],
                                                                Union[torch.tensor, type(None)],
                                                                Union[torch.tensor, type(None)],
                                                                dict]:
        # sample radiuses
        std_devs = torch.rand(self.n_functions_per_sample) * (self.noise_max - self.noise_min) + self.noise_min

        # generate example set
        example_xs = torch.randn((self.n_functions_per_sample, self.n_examples_per_sample//2, 2)) * std_devs.reshape(-1, 1, 1)

        # generate point set
        xs = torch.randn((self.n_functions_per_sample, self.n_points_per_sample//2, 2)) * std_devs.reshape(-1, 1, 1)

        # concatenate points not from this distribution
        example_xs2 = self.sample_inputs(n_functions=self.n_functions_per_sample, n_points=self.n_examples_per_sample//2,device=device, points=example_xs)
        xs2 = self.sample_inputs(n_functions=self.n_functions_per_sample, n_points=self.n_points_per_sample//2, device=device, points=xs)
        example_xs = torch.cat([example_xs, example_xs2], dim=1)
        xs = torch.cat([xs, xs2], dim=1)

        # give high logit to sampled points, and low logit to others
        example_ys = torch.zeros((self.n_functions_per_sample, self.n_examples_per_sample, 1))
        example_ys[:, :self.n_examples_per_sample//2] = self.positive_logit
        example_ys[:, self.n_examples_per_sample//2:] = self.negative_logit
        ys = torch.zeros((self.n_functions_per_sample, self.n_points_per_sample, 1))
        ys[:, :self.n_points_per_sample//2] = self.positive_logit
        ys[:, self.n_points_per_sample//2:] = self.negative_logit

        # move device
        example_xs = example_xs.to(device)
        example_ys = example_ys.to(device)
        xs = xs.to(device)
        ys = ys.to(device)
        info = {"std_devs": std_devs}
        return example_xs, example_ys, xs, ys, info

    # def sample_inputs(self, n_functions, n_points, device, points):
    #     min_distance = 0.05
    #     inputs = torch.zeros(n_functions, n_points, 2)
    #     for f in range(n_functions):
    #         count = 0
    #         while count < n_points:
    #             new_points = torch.rand(2*n_points, 2) * (self.highs - self.lows) + self.lows
    #             distances = new_points.unsqueeze(-2) - points[f].unsqueeze(-3)
    #             distances = distances.norm(dim=-1)
    #             mask = distances.min(dim=-1).values > min_distance
    #             accepted_points = new_points[mask]
    #             n_accepted = accepted_points.shape[0]
    #             inputs[f, count:min(count+n_accepted, n_points)] = accepted_points[:min(n_accepted, n_points - count)]
    #             count += n_accepted
    #     return inputs

    def sample_inputs(self, n_functions, n_points, device, points):
        points = torch.rand(n_functions, n_points, 2) * (self.highs - self.lows) + self.lows
        return points