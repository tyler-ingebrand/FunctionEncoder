from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class QuadraticDataset(BaseDataset):

    def __init__(self,
                 a_range=(-3, 3),
                 b_range=(-3, 3),
                 c_range=(-3, 3),
                 input_range=(-10, 10),
                 device: str = "auto",
                 dtype: torch.dtype = torch.float32,
                 n_examples:int=100,
                 n_queries:int=1000,
                 ):
        super().__init__(input_size=(1,),
                         output_size=(1,),
                         data_type="deterministic",
                         device=device,
                         dtype=dtype,
                         n_examples=n_examples,
                         n_queries=n_queries,
                         )
        self.a_range = torch.tensor(a_range, device=self.device, dtype=self.dtype)
        self.b_range = torch.tensor(b_range, device=self.device, dtype=self.dtype)
        self.c_range = torch.tensor(c_range, device=self.device, dtype=self.dtype)
        self.input_range = torch.tensor(input_range, device=self.device, dtype=self.dtype)

    def __getitem__(self, index) -> Tuple[  torch.tensor,
                                     torch.tensor,
                                     torch.tensor,
                                     torch.tensor,
                                     dict]:
        with torch.no_grad():
            n_examples = self.n_examples
            n_queries = self.n_queries

            # generate n_functions sets of coefficients
            As = torch.rand((1), dtype=self.dtype, device=self.device) * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
            Bs = torch.rand((1), dtype=self.dtype, device=self.device) * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
            Cs = torch.rand((1), dtype=self.dtype, device=self.device) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]

            # generate n_samples_per_function samples for each function
            query_xs = torch.rand((n_queries, *self.input_size), dtype=self.dtype, device=self.device)
            query_xs = query_xs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            example_xs = torch.rand((n_examples, *self.input_size), dtype=self.dtype, device=self.device)
            example_xs = example_xs * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

            # compute the corresponding ys
            query_ys = As.unsqueeze(1) * query_xs ** 2 + Bs.unsqueeze(1) * query_xs + Cs.unsqueeze(1)
            example_ys = As.unsqueeze(1) * example_xs ** 2 + Bs.unsqueeze(1) * example_xs + Cs.unsqueeze(1)

            return example_xs, example_ys, query_xs, query_ys, {"As":As, "Bs" : Bs, "Cs": Cs}

    def __len__(self):
        return 1000

