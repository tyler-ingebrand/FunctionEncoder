from typing import Union, Tuple
import torch
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from FunctionEncoder.Model.Architecture.Euclidean import Euclidean
from FunctionEncoder.Model.Architecture.MLP import MLP


class FunctionEncoder(torch.nn.Module):
    """A function encoder learns basis functions/vectors over a Hilbert space.

    A function encoder learns basis functions/vectors over a Hilbert space. 
    Typically, this is a function space mapping to Euclidean vectors, but it can be any Hilbert space, IE probability distributions.
    This class has a general purpose algorithm which supports both deterministic and stochastic data.
    The only difference between them is the dataset used and the inner product definition.
    This class supports two methods for computing the coefficients of the basis function, also called a representation:
    1. "inner_product": It computes the inner product of the basis functions with the data via a Monte Carlo approximation.
    2. "least_squares": This method computes the least squares solution in terms of vector operations. This typically trains faster and better. 
    This class also supports the residuals method, which learns the average function in the dataset. The residuals/error of this approximation, 
    for each function in the space, is learned via a function encoder. This tends to improve performance when the average function is not f(x) = 0. 
    """

    def __init__(self,
                 input_size:tuple[int], 
                 output_size:tuple[int], 
                 data_type:str, 
                 n_basis:int=100, 
                 model_type:Union[str, type]="MLP",
                 model_kwargs:dict=dict(),
                 method:str="least_squares", 
                 use_residuals_method:bool=False,  
                 regularization_parameter:float=1.0, 
                 gradient_accumulation:int=1,
                 ):
        """ Initializes a function encoder.

        Args:
        input_size: tuple[int]: The size of the input space, e.g. (1,) for 1D input
        output_size: tuple[int]: The size of the output space, e.g. (1,) for 1D output
        data_type: str: "deterministic" or "stochastic". Determines which defintion of inner product is used.
        n_basis: int: Number of basis functions to use.
        model_type: str: The type of model to use. See the types and kwargs in FunctionEncoder/Model/Architecture. Typically a MLP.
        model_kwargs: Union[dict, type(None)]: The kwargs to pass to the model. See the types and kwargs in FunctionEncoder/Model/Architecture.
        method: str: "inner_product" or "least_squares". Determines how to compute the coefficients of the basis functions.
        use_residuals_method: bool: Whether to use the residuals method. If True, uses an average function to predict the average of the data, and then learns the error with a function encoder.
        regularization_parameter: float: The regularization parameter for the least squares method, that encourages the basis functions to be unit length. 1 is usually good, but if your ys are very large, this may need to be increased.
        gradient_accumulation: int: The number of batches to accumulate gradients over. Typically its best to have n_functions>=10 or so, and have gradient_accumulation=1. However, sometimes due to memory reasons, or because the functions do not have the same amount of data, its necesary for n_functions=1 and gradient_accumulation>=10.
        """

        assert len(input_size) == 1, "Only 1D input supported for now"
        assert input_size[0] >= 1, "Input size must be at least 1"
        assert len(output_size) == 1, "Only 1D output supported for now"
        assert output_size[0] >= 1, "Output size must be at least 1"
        assert data_type in ["deterministic", "stochastic", "categorical"], f"Unknown data type: {data_type}"
        super(FunctionEncoder, self).__init__()
        
        # hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.method = method
        self.data_type = data_type
        
        # models and optimizers
        self.model = self._build(model_type, model_kwargs)
        self.average_function = self._build(model_type, model_kwargs, average_function=True) if use_residuals_method else None
        params = [*self.model.parameters()]
        if self.average_function is not None:
            params += [*self.average_function.parameters()]
        self.opt = torch.optim.Adam(params, lr=1e-3)

        # regulation only used for LS method
        self.regularization_parameter = regularization_parameter
        # accumulates gradients over multiple batches, typically used when n_functions=1 for memory reasons. 
        self.gradient_accumulation = gradient_accumulation

        # for printing
        self.model_type = model_type
        self.model_kwargs = model_kwargs


    def _build(self, 
               model_type:Union[str, type],
               model_kwargs:dict, 
               average_function:bool=False) -> torch.nn.Module:
        """Builds a function encoder as a single model. Can also build the average function. 
        
        Args:
        model_type: Union[str, type]: The type of model to use. See the types and kwargs in FunctionEncoder/Model/Architecture. Typically "MLP", can also be a custom class.
        model_kwargs: dict: The kwargs to pass to the model. See the kwargs in FunctionEncoder/Model/Architecture/.
        average_function: bool: Whether to build the average function. If True, builds a single function model.

        Returns:
        torch.nn.Module: The basis functions or average function model.
        """

        # the average function is basically a single function, so n_basis is 1 
        if average_function:
            n_basis = 1
        else:
            n_basis = self.n_basis

        # if provided as a string, parse the string into a class
        if type(model_type) == str:
            if model_type == "MLP":
                return MLP(input_size=self.input_size,
                           output_size=self.output_size,
                           n_basis=n_basis,
                           **model_kwargs)
            elif model_type == "Euclidean":
                return Euclidean(input_size=self.input_size,
                                 output_size=self.output_size,
                                 n_basis=n_basis,
                                 **model_kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:  # otherwise, assume it is a class and directly instantiate it
            return model_type(input_size=self.input_size,
                              output_size=self.output_size,
                              n_basis=n_basis,
                              **model_kwargs)

    def compute_representation(self, 
                               example_xs:torch.tensor, 
                               example_ys:torch.tensor, 
                               method:str="inner_product", 
                               **kwargs) -> Tuple[torch.tensor, Union[torch.tensor, None]]:
        """Computes the coefficients of the basis functions.

        This method does the forward pass of the basis functions (and the average function if it exists) over the example data.
        Then it computes the coefficients of the basis functions via a Monte Carlo integration of the inner product with the example data.
        
        Args:
        example_xs: torch.tensor: The input data. Shape (n_example_datapoints, input_size) or (n_functions, n_example_datapoints, input_size)
        example_ys: torch.tensor: The output data. Shape (n_example_datapoints, output_size) or (n_functions, n_example_datapoints, output_size)
        method: str: "inner_product" or "least_squares". Determines how to compute the coefficients of the basis functions.
        kwargs: dict: Additional kwargs to pass to the least squares method.

        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis) or (n_basis,) if n_functions=1. 
        Union[torch.tensor, None]: The gram matrix if using least squares method. None otherwise.
        """
        
        assert example_xs.shape[-len(self.input_size):] == self.input_size, f"example_xs must have shape (..., {self.input_size}). Expected {self.input_size}, got {example_xs.shape[-len(self.input_size):]}"
        assert example_ys.shape[-len(self.output_size):] == self.output_size, f"example_ys must have shape (..., {self.output_size}). Expected {self.output_size}, got {example_ys.shape[-len(self.output_size):]}"
        assert example_xs.shape[:-len(self.input_size)] == example_ys.shape[:-len(self.output_size)], f"example_xs and example_ys must have the same shape except for the last {len(self.input_size)} dimensions. Expected {example_xs.shape[:-len(self.input_size)]}, got {example_ys.shape[:-len(self.output_size)]}"

        # if not in terms of functions, add a function batch dimension
        reshaped = False
        if len(example_xs.shape) - len(self.input_size) == 1:
            reshaped = True
            example_xs = example_xs.unsqueeze(0)
            example_ys = example_ys.unsqueeze(0)

        # optionally subtract average function if we are using residuals method
        # we dont want to backprop to the average function. So we block grads. 
        if self.average_function is not None:
            with torch.no_grad():
                example_y_hat_average = self.average_function.forward(example_xs)
                example_ys = example_ys - example_y_hat_average

        # compute representation
        Gs = self.model.forward(example_xs) # forward pass of the basis functions
        if method == "inner_product":
            representation = self._compute_inner_product_representation(Gs, example_ys)
            gram = None
        elif method == "least_squares":
            representation, gram = self._compute_least_squares_representation(Gs, example_ys, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # reshape if necessary
        if reshaped:
            assert representation.shape[0] == 1, "Expected a single function batch dimension"
            representation = representation.squeeze(0)
        return representation, gram

    def _deterministic_inner_product(self, 
                                     fs:torch.tensor, 
                                     gs:torch.tensor,) -> torch.tensor:
        """Approximates the L2 inner product between fs and gs using a Monte Carlo approximation.
        Latex: \langle f, g \rangle = \frac{1}{V}\int_X f(x)g(x) dx \approx \frac{1}{n} \sum_{i=1}^n f(x_i)g(x_i)
        Note we are scaling the L2 inner product by 1/volume, which removes volume from the monte carlo approximation.
        Since scaling an inner product is still a valid inner product, this is still an inner product.
        
        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True

        # compute inner products via MC integration
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _stochastic_inner_product(self, 
                                  fs:torch.tensor, 
                                  gs:torch.tensor,) -> torch.tensor:
        """ Approximates the logit version of the inner product between continuous distributions. 
        Latex: \langle f, g \rangle = \int_X (f(x) - \Bar{f}(x) )(g(x) - \Bar{g}(x)) dx \approx \frac{1}{n} \sum_{i=1}^n (f(x_i) - \Bar{f}(x_i))(g(x_i) - \Bar{g}(x_i))
        
        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2] == 1, f"Expected fs and gs to have the same output size, which is 1 for the stochastic case since it learns the pdf(x), got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

        # compute means and subtract them
        mean_f = torch.mean(fs, dim=1, keepdim=True)
        mean_g = torch.mean(gs, dim=1, keepdim=True)
        fs = fs - mean_f
        gs = gs - mean_g

        # compute inner products
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)
        # Technically we should multiply by volume, but we are assuming that the volume is 1 since it is often not known

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _categorical_inner_product(self,
                                   fs:torch.tensor,
                                   gs:torch.tensor,) -> torch.tensor:
        """ Approximates the inner product between discrete conditional probability distributions.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, n_categories, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, n_categories, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3, 4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3, 4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, which is the number of categories in this case, got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

        # compute means and subtract them
        mean_f = torch.mean(fs, dim=2, keepdim=True)
        mean_g = torch.mean(gs, dim=2, keepdim=True)
        fs = fs - mean_f
        gs = gs - mean_g

        # compute inner products
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _inner_product(self, 
                       fs:torch.tensor, 
                       gs:torch.tensor) -> torch.tensor:
        """ Computes the inner product between fs and gs. This passes the data to either the deterministic or stochastic inner product methods.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, got {fs.shape[2]} and {gs.shape[2]}"

        if self.data_type == "deterministic":
            return self._deterministic_inner_product(fs, gs)
        elif self.data_type == "stochastic":
            return self._stochastic_inner_product(fs, gs)
        elif self.data_type == "categorical":
            return self._categorical_inner_product(fs, gs)
        else:
            raise ValueError(f"Unknown data type: '{self.data_type}'. Should be 'deterministic', 'stochastic', or 'categorical'")

    def _norm(self, fs:torch.tensor, squared=False) -> torch.tensor:
        """ Computes the norm of fs according to the chosen inner product.

        Args:
        fs: torch.tensor: The function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)

        Returns:
        torch.tensor: The Hilbert norm of fs.
        """
        norm_squared = self._inner_product(fs, fs)
        if not squared:
            return norm_squared.sqrt()
        else:
            return norm_squared

    def _distance(self, fs:torch.tensor, gs:torch.tensor, squared=False) -> torch.tensor:
        """ Computes the distance between fs and gs according to the chosen inner product.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)
        gs: torch.tensor: The second set of function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)
        returns:
        torch.tensor: The distance between fs and gs.
        """
        return self._norm(fs - gs, squared=squared)

    def _compute_inner_product_representation(self, 
                                              Gs:torch.tensor, 
                                              example_ys:torch.tensor) -> torch.tensor:
        """ Computes the coefficients via the inner product method.

        Args:
        Gs: torch.tensor: The basis functions. Shape (n_functions, n_datapoints, output_size, n_basis)
        example_ys: torch.tensor: The output data. Shape (n_functions, n_datapoints, output_size)

        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        """
        
        assert len(Gs.shape)== 4, f"Expected Gs to have shape (f,d,m,k), got {Gs.shape}"
        assert len(example_ys.shape) == 3, f"Expected example_ys to have shape (f,d,m), got {example_ys.shape}"
        assert Gs.shape[0] == example_ys.shape[0], f"Expected Gs and example_ys to have the same number of functions, got {Gs.shape[0]} and {example_ys.shape[0]}"
        assert Gs.shape[1] == example_ys.shape[1], f"Expected Gs and example_ys to have the same number of datapoints, got {Gs.shape[1]} and {example_ys.shape[1]}"
        assert Gs.shape[2] == example_ys.shape[2], f"Expected Gs and example_ys to have the same output size, got {Gs.shape[2]} and {example_ys.shape[2]}"

        # take inner product with Gs, example_ys
        inner_products = self._inner_product(Gs, example_ys)
        return inner_products

    def _compute_least_squares_representation(self, 
                                              Gs:torch.tensor, 
                                              example_ys:torch.tensor, 
                                              lambd:Union[float, type(None)]= None) -> Tuple[torch.tensor, torch.tensor]:
        """ Computes the coefficients via the least squares method.
        
        Args:
        Gs: torch.tensor: The basis functions. Shape (n_functions, n_datapoints, output_size, n_basis)
        example_ys: torch.tensor: The output data. Shape (n_functions, n_datapoints, output_size)
        lambd: float: The regularization parameter. None by default. If None, scales with 1/n_datapoints.
        
        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        torch.tensor: The gram matrix. Shape (n_functions, n_basis, n_basis)
        """
        
        assert len(Gs.shape)== 4, f"Expected Gs to have shape (f,d,m,k), got {Gs.shape}"
        assert len(example_ys.shape) == 3, f"Expected example_ys to have shape (f,d,m), got {example_ys.shape}"
        assert Gs.shape[0] == example_ys.shape[0], f"Expected Gs and example_ys to have the same number of functions, got {Gs.shape[0]} and {example_ys.shape[0]}"
        assert Gs.shape[1] == example_ys.shape[1], f"Expected Gs and example_ys to have the same number of datapoints, got {Gs.shape[1]} and {example_ys.shape[1]}"
        assert Gs.shape[2] == example_ys.shape[2], f"Expected Gs and example_ys to have the same output size, got {Gs.shape[2]} and {example_ys.shape[2]}"
        assert lambd is None or lambd >= 0, f"Expected lambda to be non-negative or None, got {lambd}"

        # set lambd to decrease with more data
        if lambd is None:
            lambd = 1/(Gs.shape[1])

        # compute gram
        gram = self._inner_product(Gs, Gs)
        gram_reg = gram + lambd * torch.eye(self.n_basis, device=gram.device)

        # compute the matrix G^TF
        ip_representation = self._inner_product(Gs, example_ys)

        # Compute (G^TG)^-1 G^TF
        ls_representation = torch.einsum("fkl,fl->fk", gram_reg.inverse(), ip_representation) # this is just batch matrix multiplication
        return ls_representation, gram

    def predict(self, 
                xs:torch.tensor,
                representations:torch.tensor, 
                precomputed_average_ys:Union[torch.tensor, None]=None) -> torch.tensor:
        """ Predicts the output of the function encoder given the input data and the coefficients of the basis functions. Uses the average function if it exists.

        Args:
        xs: torch.tensor: The input data. Shape (n_functions, n_datapoints, input_size)
        representations: torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        precomputed_average_ys: Union[torch.tensor, None]: The average function output. If None, computes it. Shape (n_functions, n_datapoints, output_size)
        
        Returns:
        torch.tensor: The predicted output. Shape (n_functions, n_datapoints, output_size)
        """

        assert len(xs.shape) == 3, f"Expected xs to have shape (f,d,n), got {xs.shape}"
        assert len(representations.shape) == 2, f"Expected representations to have shape (f,k), got {representations.shape}"
        assert xs.shape[0] == representations.shape[0], f"Expected xs and representations to have the same number of functions, got {xs.shape[0]} and {representations.shape[0]}"

        # this is weighted combination of basis functions
        Gs = self.model.forward(xs)
        y_hats = torch.einsum("fdmk,fk->fdm", Gs, representations)
        
        # optionally add the average function
        # it is allowed to be precomputed, which is helpful for training
        # otherwise, compute it
        if self.average_function:
            if precomputed_average_ys is not None:
                average_ys = precomputed_average_ys
            else:
                average_ys = self.average_function.forward(xs)
            y_hats = y_hats + average_ys
        return y_hats

    def predict_from_examples(self, 
                              example_xs:torch.tensor, 
                              example_ys:torch.tensor, 
                              xs:torch.tensor, 
                              method:str="inner_product",
                              **kwargs):
        """ Predicts the output of the function encoder given the input data and the example data. Uses the average function if it exists.
        
        Args:
        example_xs: torch.tensor: The example input data used to compute a representation. Shape (n_example_datapoints, input_size)
        example_ys: torch.tensor: The example output data used to compute a representation. Shape (n_example_datapoints, output_size)
        xs: torch.tensor: The input data. Shape (n_functions, n_datapoints, input_size)
        method: str: "inner_product" or "least_squares". Determines how to compute the coefficients of the basis functions.
        kwargs: dict: Additional kwargs to pass to the least squares method.

        Returns:
        torch.tensor: The predicted output. Shape (n_functions, n_datapoints, output_size)
        """

        assert len(example_xs.shape) == 3, f"Expected example_xs to have shape (f,d,n), got {example_xs.shape}"
        assert len(example_ys.shape) == 3, f"Expected example_ys to have shape (f,d,m), got {example_ys.shape}"
        assert len(xs.shape) == 3, f"Expected xs to have shape (f,d,n), got {xs.shape}"
        assert example_xs.shape[-len(self.input_size):] == self.input_size, f"Expected example_xs to have shape (..., {self.input_size}), got {example_xs.shape[-1]}"
        assert example_ys.shape[-len(self.input_size):] == self.output_size, f"Expected example_ys to have shape (..., {self.output_size}), got {example_ys.shape[-1]}"
        assert xs.shape[-len(self.input_size):] == self.input_size, f"Expected xs to have shape (..., {self.input_size}), got {xs.shape[-1]}"
        assert example_xs.shape[0] == example_ys.shape[0], f"Expected example_xs and example_ys to have the same number of functions, got {example_xs.shape[0]} and {example_ys.shape[0]}"
        assert example_xs.shape[1] == example_xs.shape[1], f"Expected example_xs and example_ys to have the same number of datapoints, got {example_xs.shape[1]} and {example_ys.shape[1]}"
        assert example_xs.shape[0] == xs.shape[0], f"Expected example_xs and xs to have the same number of functions, got {example_xs.shape[0]} and {xs.shape[0]}"

        representations, _ = self.compute_representation(example_xs, example_ys, method=method, **kwargs)
        y_hats = self.predict(xs, representations)
        return y_hats

    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    progress_bar=True,
                    callback:BaseCallback=None):
        """ Trains the function encoder on the dataset for some number of epochs.
        
        Args:
        dataset: BaseDataset: The dataset to train on.
        epochs: int: The number of epochs to train for.
        progress_bar: bool: Whether to show a progress bar.
        callback: BaseCallback: A callback to use during training. Can be used to test loss, etc. 
        
        Returns:
        list[float]: The losses at each epoch."""

        # verify dataset is correct
        dataset.check_dataset()
        
        # set device
        device = next(self.parameters()).device

        # Let callbacks few starting data
        if callback is not None:
            callback.on_training_start(locals())

        # method to use for representation during training
        assert self.method in ["inner_product", "least_squares"], f"Unknown method: {self.method}"

        losses = []
        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            example_xs, example_ys, xs, ys, _ = dataset.sample(device=device)

            # train average function, if it exists
            if self.average_function is not None:
                # predict averages
                expected_yhats = self.average_function.forward(xs)

                # compute average function loss
                average_function_loss = self._distance(expected_yhats, ys, squared=True).mean()
                
                # we only train average function to fit data in general, so block backprop from the basis function loss
                expected_yhats = expected_yhats.detach()
            else:
                expected_yhats = None

            # approximate functions, compute error
            representation, gram = self.compute_representation(example_xs, example_ys, method=self.method)
            y_hats = self.predict(xs, representation, precomputed_average_ys=expected_yhats)
            prediction_loss = self._distance(y_hats, ys, squared=True).mean()

            # LS requires regularization since it does not care about the scale of basis
            # so we force basis to move towards unit norm. They dont actually need to be unit, but this prevents them
            # from going to infinity.
            if self.method == "least_squares":
                norm_loss = ((torch.diagonal(gram, dim1=1, dim2=2) - 1)**2).mean()

            # add loss components
            loss = prediction_loss
            if self.method == "least_squares":
                loss = loss + self.regularization_parameter * norm_loss
            if self.average_function is not None:
                loss = loss + average_function_loss
            
            # backprop with gradient clipping
            loss.backward()
            if (epoch+1) % self.gradient_accumulation == 0:
                norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.opt.step()
                self.opt.zero_grad()

            # callbacks
            if callback is not None:
                callback.on_step(locals())

        # let callbacks know its done
        if callback is not None:
            callback.on_training_end(locals())

    def _param_string(self):
        """ Returns a dictionary of hyperparameters for logging."""
        params = {}
        params["input_size"] = self.input_size
        params["output_size"] = self.output_size
        params["n_basis"] = self.n_basis
        params["method"] = self.method
        params["model_type"] = self.model_type
        params["regularization_parameter"] = self.regularization_parameter
        for k, v in self.model_kwargs.items():
            params[k] = v
        params = {k: str(v) for k, v in params.items()}
        return params
