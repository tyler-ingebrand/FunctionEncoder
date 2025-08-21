from types import NoneType
from typing import Tuple, Union

import torch

from FunctionEncoder import BaseDataset


class FixedDataset(BaseDataset):
    def __init__(self,
                 xs:torch.tensor,
                 ys:torch.tensor,
                 n_examples:int=None,
                 n_queries:int=None,
                 device: str = "auto",
                 ):
        assert len(xs.shape) == 3, f"xs must be a 3D tensor, the first dim corresponds to functions, the second to data points, the third to input size. Got shape: {xs.shape}"
        assert len(ys.shape) == 3, f"ys must be a 3D tensor, the first dim corresponds to functions, the second to data points, the third to output size. Got shape: {ys.shape}"
        assert xs.shape[0] == ys.shape[0], f"xs and ys must have the same number of functions. Got xs.shape[0]: {xs.shape[0]}, ys.shape[0]: {ys.shape[0]}"
        assert xs.shape[1] == ys.shape[1], f"xs and ys must have the same number of data points. Got xs.shape[1]: {xs.shape[1]}, ys.shape[1]: {ys.shape[1]}"
        super().__init__(input_size=(xs.shape[2],),
                         output_size=(ys.shape[2],),
                         data_type="deterministic",
                         device=device,
                         n_examples=n_examples,
                         n_queries=n_queries,
                         )
        self.xs = xs.to(self.device)
        self.ys = ys.to(self.device)


    def __getitem__(self, index) -> Tuple[ torch.tensor,
                                            torch.tensor,
                                            torch.tensor,
                                            torch.tensor,
                                            dict]:
        example_indicies = torch.randint(0, self.xs.shape[1], (self.n_examples,), device=self.device)
        point_indicies = torch.randint(0, self.xs.shape[1], (self.n_queries,), device=self.device)
        examples_xs = self.xs[index][example_indicies]
        examples_ys = self.ys[index][ example_indicies]
        query_xs = self.xs[index][point_indicies]
        query_ys = self.ys[index][ point_indicies]

        info = {} # nothing interesting here
        return examples_xs, examples_ys, query_xs, query_ys, info


