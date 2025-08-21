import torch
from typing import Union
from .MLP import MLP
from .ParallelMLP import ParallelMLP
from .Euclidean import Euclidean
from .CNN import CNN
from .NeuralODE import NeuralODE


# Feel free to append to this list to implement custom architectures.
# Or, simply pass in the new model class directly to the build function.
SUPPORTED_ARCHITECTURES = {
    "MLP": MLP,
    "ParallelMLP": ParallelMLP,
    "Euclidean": Euclidean,
    "CNN": CNN,
    "NeuralODE": NeuralODE,
}


def build(
        input_size: tuple[int],
        output_size: tuple[int],
        n_basis,
        model_type: Union[str, type],
        model_kwargs: dict,
        average_function: bool = False,
) -> torch.nn.Module:
    """Builds a function encoder as a single model. Can also build the average function.

    Args:
    model_type: Union[str, type]: The type of model to use. See the types and kwargs in FunctionEncoder/Model/Architecture. Typically "MLP", can also be a custom class.
    model_kwargs: dict: The kwargs to pass to the model. See the kwargs in FunctionEncoder/Model/Architecture/.
    average_function: bool: Whether to build the average function. If True, builds a single function model.

    Returns:
    torch.nn.Module: The basis functions or average function model.
    """
    assert type(model_type) in [str, type], \
        f"model_type must be a string or a class. Got {type(model_type)}. Supported architectures: {SUPPORTED_ARCHITECTURES.keys()}"
    if type(model_type) == str:
        assert model_type in SUPPORTED_ARCHITECTURES, f"model_type must be one of {SUPPORTED_ARCHITECTURES.keys()}. Got {model_type}."
        model_type = SUPPORTED_ARCHITECTURES[model_type]

    # Create the model
    model = model_type(
        input_size=input_size,
        output_size=output_size,
        n_basis=n_basis,
        average_function=average_function,
        **model_kwargs
    )

    return model


def predict_number_params(input_size: tuple[int],
                          output_size: tuple[int],
                          n_basis: int = 100,
                          model_type: Union[str, type] = "MLP",
                          model_kwargs: dict = dict(),
                          use_residuals_method: bool = False
                          ):
    """ Predicts the number of parameters in the function encoder.
    Useful for ensuring all experiments use the same number of params"""
    assert type(model_type) in [str, type], \
        f"model_type must be a string or a class. Got {type(model_type)}. Supported architectures: {SUPPORTED_ARCHITECTURES.keys()}"
    if type(model_type) == str:
        assert model_type in SUPPORTED_ARCHITECTURES, f"model_type must be one of {SUPPORTED_ARCHITECTURES.keys()}. Got {model_type}."
        model_type = SUPPORTED_ARCHITECTURES[model_type]

    n_params = model_type.predict_number_params(input_size, output_size, n_basis, average_function=False, **model_kwargs)
    if use_residuals_method:
        n_params += model_type.predict_number_params(input_size, output_size, n_basis, average_function=True, **model_kwargs)

    return n_params