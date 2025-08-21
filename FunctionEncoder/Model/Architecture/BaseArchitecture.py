import torch


class BaseArchitecture(torch.nn.Module):

    def __init__(self,
                 input_size: tuple[int],
                 output_size: tuple[int],
                 n_basis: int,
                 average_function: bool = False,
                 ):
        """
        Base class for all architectures in FunctionEncoder.
        This class should not be instantiated directly.
        Instead, use one of the subclasses like MLP, ParallelMLP, NeuralODE, etc.
        :param input_size: tuple of input size
        :param output_size: tuple of output size
        :param n_basis: number of basis functions
        :param average_function: If true, the model is meant to learn the average function. So its output
            does not return a basis. Example:
            average_function=False => f: R^n -> R^{m,k}
            average_function=True  => f: R^n -> R^m
        """
        super(BaseArchitecture, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.average_function = average_function


    @staticmethod
    def predict_number_params(*args, **kwargs):
        raise NotImplementedError()