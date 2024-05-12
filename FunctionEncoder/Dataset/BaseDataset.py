import torch
from abc import abstractmethod
from typing import Tuple, Union


class BaseDataset:

    def __init__(self,
                 input_size:Tuple[int],
                 output_size:Tuple[int],
                 total_n_functions:Union[int, float],
                 total_n_samples_per_function:Union[int, float],
                 data_type:str,
                 n_functions_per_sample:int,
                 n_examples_per_sample:int,
                 n_points_per_sample:int,
                 ):
        assert len(input_size) >= 1, "input_size must be a tuple of at least one element"
        assert len(output_size) >= 1, "output_size must be a tuple of at least one element"
        assert total_n_functions >= 1, "n_functions must be a positive integer or infinite"
        assert total_n_samples_per_function >= 1, "n_samples_per_function must be a positive integer or infinite"
        assert data_type in ["deterministic", "stochastic"]
        self.input_size = input_size
        self.output_size = output_size
        self.n_functions = total_n_functions # may be infinite
        self.n_samples_per_function = total_n_samples_per_function # may be infinite
        self.data_type = data_type
        self.n_functions_per_sample = n_functions_per_sample
        self.n_examples_per_sample = n_examples_per_sample
        self.n_points_per_sample = n_points_per_sample

    @abstractmethod
    def sample(self, device:Union[str, torch.device]) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, dict]:
        pass