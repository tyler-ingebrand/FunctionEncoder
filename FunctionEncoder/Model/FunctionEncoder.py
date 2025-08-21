# globals
from typing import Union, Tuple
import torch
from tqdm import trange

# locals
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.build import build, predict_number_params
from FunctionEncoder.Model.InnerProduct import INNER_PRODUCTS


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
                 regularization_parameter:float=1.0, # if you normalize your data, this is usually good
                 gradient_accumulation:int=1, # default: no gradient accumulation
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs:dict={"lr":1e-3},
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
        optimizer: torch.optim.Optimizer: The optimizer to use for training the model. Defaults to Adam.
        optimizer_kwargs: dict: The kwargs to pass to the optimizer. Defaults to {"lr": 1e-3}.
        """

        if model_type == "MLP":
            assert len(input_size) == 1, "MLP only supports 1D input"
        if model_type == "ParallelMLP":
            assert len(input_size) == 1, "ParallelMLP only supports 1D input"
        if model_type == "CNN":
            assert len(input_size) == 3, "CNN only supports 3D input"
        if isinstance(model_type, type):
            assert issubclass(model_type, BaseArchitecture), "model_type should be a subclass of BaseArchitecture. This just gives a way of predicting the number of parameters before init."
        assert len(input_size) in [1, 3], "Input must either be 1-Dimensional (euclidean vector) or 3-Dimensional (image)"
        assert input_size[0] >= 1, "Input size must be at least 1"
        assert len(output_size) == 1, "Only 1D output supported for now"
        assert output_size[0] >= 1, "Output size must be at least 1"
        assert data_type in INNER_PRODUCTS.keys(), f"Unknown data type: {data_type}. Options are {INNER_PRODUCTS.keys()}."
        super(FunctionEncoder, self).__init__()
        
        # hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.method = method
        self.data_type = data_type

        # get the correct inner product
        self._inner_product = INNER_PRODUCTS[self.data_type]

        # Create basis functions and average function.
        self.basis_functions = build(
            input_size=input_size,
            output_size=output_size,
            n_basis=n_basis,
            model_type=model_type,
            model_kwargs=model_kwargs,
            average_function=False,
        )
        self.average_function = build(
            input_size=input_size,
            output_size=output_size,
            n_basis=1,  # average function is always a single function
            model_type=model_type,
            model_kwargs=model_kwargs,
            average_function=True,
        ) if use_residuals_method else None

        # Create optimizer.
        params = list(self.basis_functions.parameters())
        if self.average_function is not None:
            params += list(self.average_function.parameters())
        self.opt = optimizer(params, **optimizer_kwargs) # usually ADAM with lr 1e-3

        # regulation only used for LS method to ensure that the basis functions do not grow infinitely.
        # this is not used if you calculate coefficients using the inner product method only.
        self.regularization_parameter = regularization_parameter

        # accumulates gradients over multiple batches, typically used when n_functions must be 1 for memory reasons.
        self.gradient_accumulation = gradient_accumulation

        # for printing
        self.model_type = model_type
        self.model_kwargs = model_kwargs

        # verify number of parameters
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = FunctionEncoder.predict_number_params(input_size=input_size, output_size=output_size, n_basis=n_basis, model_type=model_type, model_kwargs=model_kwargs, use_residuals_method=use_residuals_method)
        assert n_params == estimated_n_params, f"Model has {n_params} parameters, but expected {estimated_n_params} parameters."



    def compute_representation(self, 
                               example_xs:torch.tensor, 
                               example_ys:torch.tensor, 
                               method:str=None,
                               lambd=1e-3,
                               ) -> Tuple[torch.tensor, Union[torch.tensor, None]]:
        """Computes the coefficients of the basis functions.

        This method does the forward pass of the basis functions (and the average function if it exists) over the example data.
        Then it computes the coefficients of the basis functions via a Monte Carlo integration of the inner product with the example data.
        
        Args:
        example_xs: torch.tensor: The input data. Shape (d, n) or (f, d, n)
        example_ys: torch.tensor: The output data. Shape (d, *m) or (f, d, *m)
        method: str: "inner_product" or "least_squares". Determines how to compute the coefficients of the basis functions.
            Defaults to the training method.
        kwargs: dict: Additional kwargs to pass to the least squares method.

        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (k,) or (f, k)
        Union[torch.tensor, None]: The gram matrix if using least squares method. None otherwise.
        """
        
        assert example_xs.shape[-len(self.input_size):] == self.input_size, f"example_xs must have shape (..., {self.input_size}). Expected {self.input_size}, got {example_xs.shape[-len(self.input_size):]}"
        assert example_ys.shape[-len(self.output_size):] == self.output_size, f"example_ys must have shape (..., {self.output_size}). Expected {self.output_size}, got {example_ys.shape[-len(self.output_size):]}"
        assert example_xs.shape[:-len(self.input_size)] == example_ys.shape[:-len(self.output_size)], f"example_xs and example_ys must have the same shape except for the last {len(self.input_size)} dimensions. Expected {example_xs.shape[:-len(self.input_size)]}, got {example_ys.shape[:-len(self.output_size)]}"
        if method == "inner_product":
            assert self.method != "least_squares", ("Cannot train using least squares, then eval using inner product. "
                                                    "If you train using LS, the basis functions are NOT orthonormal, "
                                                    "and therefore you must use LS to correct for this. ")

        # set method to the training method if not provided
        method = method or self.method

        # optionally subtract average function if we are using residuals method
        # we dont want to backprop to the average function. So we block grads. 
        if self.average_function is not None:
            with torch.no_grad():
                example_y_hat_average = self.average_function.forward(example_xs)
                example_ys = example_ys - example_y_hat_average

        # Get basis outputs
        Gs = self.forward_basis_functions(example_xs)

        # Compute representation.
        # if using the inner product only method, then
        #        ┌          ┐
        #        | <f, g_1> |
        #   c =  |    :     |
        #        | <f, g_k> |
        #        └          ┘
        # Note this is equivalent to LS if the gram is I without regularization
        # If using least squares, then
        #        ┌                              ┐-1    ┌          ┐
        #        | <g_1, g_1>  ...  <g_1, g_k>  |      | <f, g_1> |
        #  c =   |     :                 :      |      |    :     |
        #        | <g_k, g_1>  ...  <g_k, g_k>  |      | <f, g_k> |
        #        └                              ┘      └          ┘
        # possibly with some regularization.

        # Get the column vector.
        f_ip_gs = self.inner_product(Gs, example_ys)

        if method == "least_squares":
            # compute gram
            gram = self.inner_product(Gs, Gs)
            gram_reg = gram + lambd * torch.eye(self.n_basis, device=gram.device)

            # Compute (G^TG)^-1 G^TF
            c = torch.einsum("...kl,...l->...k", gram_reg.inverse(), f_ip_gs)  # this is just batch matrix multiplication
        else:
            gram = None
            c = f_ip_gs
        return c, gram


    def inner_product(self,
                       fs:torch.tensor, 
                       gs:torch.tensor) -> torch.tensor:
        """ Computes the inner product between fs and gs. This first reshapes the inputs to a pre-determined size, then
        calls the inner product for this data type as specified in the constructor. Then it sets the shape back
        so the output matches the data dimensions.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape (d,m), (f,d,m), or (f,d,m,k)
        gs: torch.tensor: The second set of function outputs. Shape (d,m), (f,d,m), or (f,d,m,l)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (1,) or (k,) or (f,k) or (f,l) or (f,k,l),
            depending on the input shapes.
        """

        # Check input ranks
        assert fs.ndim in [2, 3, 4], f"fs must have shape (d,m), (f,d,m), or (f,d,m,k), got {fs.shape}"
        assert gs.ndim in [2, 3, 4], f"gs must have shape (d,m), (f,d,m), or (f,d,m,l), got {gs.shape}"
        if fs.ndim == 2:
            assert gs.ndim == 2, (f"If fs is 2D, gs must also be 2D, got fs={fs.shape} and gs={gs.shape}."
                                  f"This is because the inner product depends on the distribution of inputs,"
                                  f"and often the distribution is different per function, so this may cause"
                                  f"erroneous inner product computation.")

        # Align shapes: add batch dim f if missing, and trailing feature dims if missing
        if fs.ndim == 2:  # (d,m)
            fs = fs.unsqueeze(0).unsqueeze(-1)  # -> (1,d,m,1)
        elif fs.ndim == 3:  # (f,d,m)
            fs = fs.unsqueeze(-1)  # -> (f,d,m,1)

        if gs.ndim == 2:  # (d,m)
            gs = gs.unsqueeze(0).unsqueeze(-1)  # -> (1,d,m,1)
        elif gs.ndim == 3:  # (f,d,m)
            gs = gs.unsqueeze(-1)  # -> (f,d,m,1)

        # Verify compatible sizes
        assert fs.shape[0] == gs.shape[0], f"Mismatch in f: {fs.shape[0]} vs {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Mismatch in d: {fs.shape[1]} vs {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Mismatch in m: {fs.shape[2]} vs {gs.shape[2]}"

        # Compute inner product
        ips = self._inner_product(fs, gs)

        # Squeeze out only the added singleton dims to restore expected shapes
        return ips.squeeze()

    def _norm(self, fs:torch.tensor, squared=False) -> torch.tensor:
        """ Computes the norm of fs according to the chosen inner product.

        Args:
        fs: torch.tensor: The function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)

        Returns:
        torch.tensor: The Hilbert norm of fs.
        """
        norm_squared = self.inner_product(fs, fs)
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



    def predict(self, 
                query_xs:torch.tensor,
                representations:torch.tensor
                ) -> torch.tensor:
        """ Predicts the output of the function encoder given the input data and the coefficients of the basis functions. Uses the average function if it exists.

        Args:
        xs: torch.tensor: The input data. Shape (n_functions, n_datapoints, input_size)
        representations: torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        precomputed_average_ys: Union[torch.tensor, None]: The average function output. If None, computes it. Shape (n_functions, n_datapoints, output_size)
        
        Returns:
        torch.tensor: The predicted output. Shape (n_functions, n_datapoints, output_size)
        """

        assert len(query_xs.shape) == 2 + len(self.input_size), f"Expected xs to have shape (f,d,*n), got {query_xs.shape}"
        # assert len(representations.shape) == 2, f"Expected representations to have shape (f,k), got {representations.shape}"
        # assert query_xs.shape[0] == representations.shape[0], f"Expected xs and representations to have the same number of functions, got {query_xs.shape[0]} and {representations.shape[0]}"

        # this is weighted combination of basis functions
        Gs = self.forward_basis_functions(query_xs)
        y_hats = torch.einsum("...dmk,...k->...dm", Gs, representations)
        
        # optionally add the average function
        # it is allowed to be precomputed, which is helpful for training
        # otherwise, compute it
        if self.average_function:
            average_ys = self.forward_average_function(query_xs)
            y_hats = y_hats + average_ys
        return y_hats

    def predict_from_examples(self, 
                              example_xs:torch.tensor, 
                              example_ys:torch.tensor, 
                              query_xs:torch.tensor,
                              method:str=None,
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

        assert len(example_xs.shape) == 2 + len(self.input_size), f"Expected example_xs to have shape (f,d,*n), got {example_xs.shape}"
        assert len(example_ys.shape) == 2 + len(self.output_size), f"Expected example_ys to have shape (f,d,*m), got {example_ys.shape}"
        assert len(query_xs.shape) == 2 + len(self.input_size), f"Expected xs to have shape (f,d,*n), got {query_xs.shape}"
        assert example_xs.shape[-len(self.input_size):] == self.input_size, f"Expected example_xs to have shape (..., {self.input_size}), got {example_xs.shape[-1]}"
        assert example_ys.shape[-len(self.output_size):] == self.output_size, f"Expected example_ys to have shape (..., {self.output_size}), got {example_ys.shape[-1]}"
        assert query_xs.shape[-len(self.input_size):] == self.input_size, f"Expected xs to have shape (..., {self.input_size}), got {query_xs.shape[-1]}"
        assert example_xs.shape[0] == example_ys.shape[0], f"Expected example_xs and example_ys to have the same number of functions, got {example_xs.shape[0]} and {example_ys.shape[0]}"
        assert example_xs.shape[1] == example_xs.shape[1], f"Expected example_xs and example_ys to have the same number of datapoints, got {example_xs.shape[1]} and {example_ys.shape[1]}"
        assert example_xs.shape[0] == query_xs.shape[0], f"Expected example_xs and xs to have the same number of functions, got {example_xs.shape[0]} and {query_xs.shape[0]}"

        representations, _ = self.compute_representation(example_xs, example_ys, method=method, **kwargs)
        y_hats = self.predict(query_xs, representations)
        return y_hats


    def estimate_L2_error(self, example_xs, example_ys):
        """ Estimates the L2 error of the function encoder on the example data. 
        This gives an idea if the example data lies in the span of the basis, or not.
        
        Args:
        example_xs: torch.tensor: The example input data used to compute a representation. Shape (n_functions, n_example_datapoints, input_size)
        example_ys: torch.tensor: The example output data used to compute a representation. Shape (n_functions, n_example_datapoints, output_size)
        
        Returns:
        torch.tensor: The estimated L2 error. Shape (n_functions,)
        """
        representation, gram = self.compute_representation(example_xs, example_ys, method="least_squares")
        f_hat_norm_squared = representation @ gram @ representation.T
        f_norm_squared = self._inner_product(example_ys, example_ys)
        l2_distance = torch.sqrt(f_norm_squared - f_hat_norm_squared)
        return l2_distance



    def train_model(self,
                    dataloader: torch.utils.data.DataLoader,
                    grad_steps: int,
                    progress_bar=True,
                    callback:BaseCallback=None,
                    **kwargs):
        """ Trains the function encoder on the dataset for some number of epochs.
        
        Args:
        dataset: BaseDataset: The dataset to train on.
        epochs: int: The number of epochs to train for.
        progress_bar: bool: Whether to show a progress bar.
        callback: BaseCallback: A callback to use during training. Can be used to test loss, etc. 
        
        Returns:
        list[float]: The losses at each epoch."""

        # verify dataset is correct
        dataloader.dataset.check_dataset()
        loader_iter = iter(dataloader)
        
        # set device
        device = next(self.parameters()).device

        # Let callbacks few starting data
        if callback is not None:
            callback.on_training_start(locals())

        # method to use for representation during training
        assert self.method in ["inner_product", "least_squares"], f"Unknown method: {self.method}"

        losses = []
        bar = trange(grad_steps) if progress_bar else range(grad_steps)
        for grad_step in bar:
            # fetch data
            try:
                example_xs, example_ys, query_xs, query_ys, _ = next(loader_iter)
            except StopIteration:
                # restart iterator if we reach the end
                loader_iter = iter(dataloader)
                example_xs, example_ys, query_xs, query_ys, _ = next(loader_iter)

            # change device
            example_xs, example_ys, query_xs, query_ys = example_xs.to(device), example_ys.to(device), query_xs.to(device), query_ys.to(device)

            # train average function, if it exists
            if self.average_function is not None:
                # predict averages
                expected_yhats = self.forward_average_function(query_xs)

                # compute average function loss
                average_function_loss = self._distance(expected_yhats, query_ys, squared=True).mean()


            # approximate functions, compute error
            representation, gram = self.compute_representation(example_xs, example_ys, method=self.method, **kwargs)
            y_hats = self.predict(query_xs, representation)
            prediction_loss = self._distance(y_hats, query_ys, squared=True).mean()

            # LS requires regularization since it does not care about the scale of basis
            # so we force basis to move towards unit norm. They dont actually need to be unit, but this prevents them
            # from going to infinity.
            if self.method == "least_squares":
                norm_loss = ((torch.diagonal(gram, dim1=-2, dim2=-1) - 1)**2).mean()

            # add loss components
            loss = prediction_loss
            if self.method == "least_squares":
                loss = loss + self.regularization_parameter * norm_loss
            if self.average_function is not None:
                loss = loss + average_function_loss
            
            # backprop with gradient clipping
            loss.backward()
            if (grad_step+1) % self.gradient_accumulation == 0:
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

    @staticmethod
    def predict_number_params(input_size:tuple[int],
                             output_size:tuple[int],
                             n_basis:int=100,
                             model_type:Union[str, type]="MLP",
                             model_kwargs:dict=dict(),
                             use_residuals_method: bool = False,
                             *args, **kwargs):
        return predict_number_params(
            input_size=input_size,
            output_size=output_size,
            n_basis=n_basis,
            model_type=model_type,
            model_kwargs=model_kwargs,
            use_residuals_method=use_residuals_method,
        )

    def forward_basis_functions(self, xs:torch.tensor) -> torch.tensor:
        """ Forward pass of the basis functions. """
        Gs = self.basis_functions(xs)
        assert Gs.shape == (xs.shape[:-len(self.input_size)] + self.output_size) + (self.n_basis,), \
            (f"Expected Gs to have shape {xs.shape[:-len(self.input_size)] + (self.n_basis,) + self.output_size}, "
             f"got {Gs.shape}")
        return Gs

    def forward_average_function(self, xs:torch.tensor) -> torch.tensor:
        """ Forward pass of the average function. """
        return self.average_function(xs) if self.average_function is not None else None