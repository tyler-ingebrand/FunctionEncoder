from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class QuadraticDataset(BaseDataset):

    def __init__(self,
                 a_range=(-3, 3),
                 b_range=(-3, 3),
                 c_range=(-3, 3),
                 input_range=(-10, 10),
                 n_functions_per_sample:int = 10,
                 n_examples_per_sample:int = 1_000,
                 n_points_per_sample:int = 10_000,
                 device:str="auto",
                 dtype:torch.dtype = torch.float32,
                 ):
        super().__init__(input_size=(1,),
                         output_size=(1,),
                         total_n_functions=float('inf'),
                         total_n_samples_per_function=float('inf'),
                         data_type="deterministic",
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,
                         device=device,
                         dtype=dtype,
                         )
        self.a_range = torch.tensor(a_range, device=self.device, dtype=self.dtype)
        self.b_range = torch.tensor(b_range, device=self.device, dtype=self.dtype)
        self.c_range = torch.tensor(c_range, device=self.device, dtype=self.dtype)
        self.input_range = torch.tensor(input_range, device=self.device, dtype=self.dtype)

    def sample(self) -> Tuple[  torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                dict]:
        with torch.no_grad():
            n_functions = self.n_functions_per_sample
            n_examples = self.n_examples_per_sample
            n_points = self.n_points_per_sample

            # generate n_functions sets of coefficients
            As = torch.rand((n_functions, 1), dtype=self.dtype, device=self.device) * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
            Bs = torch.rand((n_functions, 1), dtype=self.dtype, device=self.device) * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
            Cs = torch.rand((n_functions, 1), dtype=self.dtype, device=self.device) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]

            # generate n_samples_per_function samples for each function
            xs = torch.rand((n_functions, n_points, *self.input_size), dtype=self.dtype, device=self.device)
            xs = xs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            example_xs = torch.rand((n_functions, n_examples, *self.input_size), dtype=self.dtype, device=self.device)
            example_xs = example_xs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

            # compute the corresponding ys
            ys = As.unsqueeze(1) * xs ** 2 + Bs.unsqueeze(1) * xs + Cs.unsqueeze(1)
            example_ys = As.unsqueeze(1) * example_xs ** 2 + Bs.unsqueeze(1) * example_xs + Cs.unsqueeze(1)

            return example_xs, example_ys, xs, ys, {"As":As, "Bs" : Bs, "Cs": Cs}

