# from types import NoneType

import torch
from abc import abstractmethod
from typing import Tuple, Union


class BaseDataset:
    """ Base class for all datasets. Follow this interface to interact with FunctionEncoder.py"""

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
        """ Constructor for BaseDataset
        
        Args:
        input_size (Tuple[int]): Size of input to the function space, IE the number of dimensions of the input
        output_size (Tuple[int]): Size of output of the function space, IE the number of dimensions of the output
        total_n_functions (Union[int, float]): Number of functions in this dataset. If functions are sampled from a continuous space, this can be float('inf')
        total_n_samples_per_function (Union[int, float]): Number of data points per function. If data is sampled from a continuous space, this can be float('inf')
        data_type (str): Type of data. Options are "deterministic" or "stochastic". Affects which inner product method is used. 
        n_functions_per_sample (int): Number of functions per training step. Should be at least 5 or so.
        n_examples_per_sample (int): Number of example points per function per training step. This data is used by the function encoder to compute coefficients.
        n_points_per_sample (int): Number of target points per function per training step. Should be large enough to capture the function's behavior. These points are used to train the function encoder as the target of the prediction, ie the MSE.

        """
        assert len(input_size) >= 1, "input_size must be a tuple of at least one element"
        assert len(output_size) >= 1, "output_size must be a tuple of at least one element"
        assert total_n_functions >= 1, "n_functions must be a positive integer or infinite"
        assert total_n_samples_per_function >= 1, "n_samples_per_function must be a positive integer or infinite"
        assert data_type in ["deterministic", "stochastic", "categorical"]
        self.input_size = input_size
        self.output_size = output_size
        self.n_functions = total_n_functions # may be infinite
        self.n_samples_per_function = total_n_samples_per_function # may be infinite
        self.data_type = data_type.lower()
        self.n_functions_per_sample = n_functions_per_sample
        self.n_examples_per_sample = n_examples_per_sample
        self.n_points_per_sample = n_points_per_sample

    @abstractmethod
    def sample(self, device:Union[str, torch.device]) -> Tuple[ torch.tensor,
                                                                torch.tensor,                                                                 
                                                                torch.tensor,
                                                                torch.tensor,
                                                                dict]:
        """Sample a batch of functions from the dataset.

        Args:
        device (Union[str, torch.device]): Device to put the data on. Can be a string or a torch.device object.

        Returns:
        torch.tensor: Example Input data to compute a representation. Shape is (n_functions, n_examples, input_size)
        torch.tensor: Example Output data to compute a representation. Shape is (n_functions, n_examples, output_size)
        torch.tensor: Input data to predict outputs for. Shape is (n_functions, n_points, input_size)
        torch.tensor: Output data, IE target of the prediction. Shape is (n_functions, n_points, output_size)
        """
        pass

def check_dataset(dataset:BaseDataset):
    """ Verify that a dataset is correctly implemented. Throws error if violated. """
    out = dataset.sample("cpu")
    assert len(out) == 5, f"Expected 5 outputs, got {len(out)}"
    
    example_xs, example_ys, xs, ys, info = out
    assert type(example_xs) == torch.Tensor, f"Expected example_xs to be a torch.Tensor, got {type(example_xs)}"
    assert type(example_ys) == torch.Tensor, f"Expected example_ys to be a torch.Tensor, got {type(example_ys)}"
    assert type(xs) == torch.Tensor, f"Expected xs to be a torch.Tensor, got {type(xs)}"
    assert type(ys) == torch.Tensor, f"Expected ys to be a torch.Tensor, got {type(ys)}"
    assert example_xs.shape == (dataset.n_functions_per_sample, dataset.n_examples_per_sample, *dataset.input_size), f"Expected example_xs shape to be {(dataset.n_functions_per_sample, dataset.n_examples_per_sample, *dataset.input_size)}, got {example_xs.shape}"
    assert example_ys.shape == (dataset.n_functions_per_sample, dataset.n_examples_per_sample, *dataset.output_size), f"Expected example_ys shape to be {(dataset.n_functions_per_sample, dataset.n_examples_per_sample, *dataset.output_size)}, got {example_ys.shape}"
    assert xs.shape == (dataset.n_functions_per_sample, dataset.n_points_per_sample, *dataset.input_size), f"Expected xs shape to be {(dataset.n_functions_per_sample, dataset.n_points_per_sample, *dataset.input_size)}, got {xs.shape}"
    assert ys.shape == (dataset.n_functions_per_sample, dataset.n_points_per_sample, *dataset.output_size), f"Expected ys shape to be {(dataset.n_functions_per_sample, dataset.n_points_per_sample, *dataset.output_size)}, got {ys.shape}"
    assert type(info) == dict, f"Expected info to be a dict, got {type(info)}"