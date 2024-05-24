from typing import Union, Tuple

import torch

from FunctionEncoder import BaseDataset



# this class samples points in the xy plane
class EuclideanDataset(BaseDataset):
    def __init__(self):
        input_size = (1,)
        output_size = (3,)
        data_type = "deterministic"
        total_n_functions = float("inf")
        total_n_samples_per_function = 1
        n_functions_per_sample = 10
        n_examples_per_sample = 1
        n_points_per_sample = 1
        super(EuclideanDataset, self).__init__(input_size=input_size,
                                              output_size=output_size,
                                              total_n_functions=total_n_functions,
                                              total_n_samples_per_function=total_n_samples_per_function,
                                              data_type=data_type,
                                              n_functions_per_sample=n_functions_per_sample,
                                              n_examples_per_sample=n_examples_per_sample,
                                              n_points_per_sample=n_points_per_sample)
        self.min = torch.tensor([-1, -1, 0])
        self.max = torch.tensor([1, 1, 0])

    def sample(self, device: Union[str, torch.device]) -> Tuple[Union[torch.tensor, type(None)],
                                                                Union[torch.tensor, type(None)],
                                                                Union[torch.tensor, type(None)],
                                                                Union[torch.tensor, type(None)],
                                                                dict]:
        # these are unused, except for the size
        example_xs = torch.zeros(self.n_functions_per_sample, self.n_examples_per_sample, *self.input_size)
        xs = torch.zeros(self.n_functions_per_sample, self.n_points_per_sample, *self.input_size)

        # sample the ys
        example_ys = torch.rand(self.n_functions_per_sample, self.n_examples_per_sample, *self.output_size) * (self.max - self.min) + self.min
        ys = example_ys

        # change device
        example_xs = example_xs.to(device)
        example_ys = example_ys.to(device)
        xs = xs.to(device)
        ys = ys.to(device)
        return example_xs, example_ys, xs, ys, {}