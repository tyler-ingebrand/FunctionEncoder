from typing import Union, Tuple

import torch

from FunctionEncoder import BaseDataset



# this class samples points in the xy plane
class EuclideanDataset(BaseDataset):
    def __init__(self):
        input_size = (1,)
        output_size = (3,)
        data_type = "deterministic"
        device = "cpu"
        super(EuclideanDataset, self).__init__(input_size=input_size,
                                              output_size=output_size,
                                              data_type=data_type,
                                              n_examples=1,
                                              n_queries=1,
                                              device=device)
        self.min = torch.tensor([-1, -1, 0])
        self.max = torch.tensor([1, 1, 0])

    def __getitem__(self, item) -> Tuple[torch.tensor,
                                                                torch.tensor, 
                                                                torch.tensor, 
                                                                torch.tensor, 
                                                                dict]:
        # these are unused, except for the size
        example_xs = torch.zeros( self.n_examples, *self.input_size)
        query_xs = torch.zeros( self.n_queries, *self.input_size)

        # sample the ys
        example_ys = torch.rand( self.n_examples, *self.output_size) * (self.max - self.min) + self.min
        query_ys = example_ys

        return example_xs, example_ys, query_xs, query_ys, {}

    def __len__(self):
        return 1000