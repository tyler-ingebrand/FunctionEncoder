import math
from typing import Union

import torch

from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.MLP import get_activation, MLP


def rk4_difference_only(model, xs, dt):
    """
    Runge-Kutta 4th order method for solving ODEs.
    WARNING: This method does not append the initial states to the output, so it only predicts the change in state
    :param model: the model to use for the ODE
    :param xs: the input data
    :param dt: the time step
    :return: the output of the model
    """
    k1 = model(xs)
    k2 = model(xs + dt / 2 * k1)
    k3 = model(xs + dt / 2 * k2)
    k4 = model(xs + dt * k3)
    return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)



class NeuralODE(BaseArchitecture):
    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size=77, n_layers=4, learn_basis_functions=True, *args, **kwargs):
        input_size = input_size[0]
        output_size = output_size[0]
        # +1 accounts for bias
        n_params =  (input_size+1) * hidden_size + \
                    (hidden_size+1) * hidden_size * (n_layers - 2) + \
                    (hidden_size+1) * output_size
        if learn_basis_functions:
            n_params *= n_basis
        return n_params


    def __init__(self,
                 input_size:tuple[int],
                 output_size:tuple[int],
                 n_basis:int=100,
                 hidden_size:int=77,
                 n_layers:int=4,
                 activation:str="relu",
                 learn_basis_functions=True,
                 dt:float=0.1,):
        super(NeuralODE, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.learn_basis_functions = learn_basis_functions
        self.dt = dt

        if not self.learn_basis_functions:
            n_basis = 1
            self.n_basis = 1


        # build net
        self.models = torch.nn.ModuleList([
            MLP(input_size=input_size,
              output_size=output_size,
              n_basis=1,
              hidden_size=hidden_size,
              n_layers=n_layers,
              activation=activation,
              learn_basis_functions=False,
              )
            for _ in range(n_basis)]
        )

        # verify number of parameters
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = self.predict_number_params(self.input_size, self.output_size, n_basis, hidden_size, n_layers, learn_basis_functions=learn_basis_functions)
        assert n_params == estimated_n_params, f"Model has {n_params} parameters, but expected {estimated_n_params} parameters."


    def forward(self, x):
        assert x.shape[-1] == self.input_size[0], f"Expected input size {self.input_size[0]}, got {x.shape[-1]}"
        reshape = None
        if len(x.shape) == 1:
            reshape = 1
            x = x.reshape(1, 1, -1)
        if len(x.shape) == 2:
            reshape = 2
            x = x.unsqueeze(0)

        # this is the main part of this function. The rest is just error handling
        outs = [rk4_difference_only(model, x, self.dt) for model in self.models]
        Gs = torch.stack(outs, dim=-1)
        if not self.learn_basis_functions:
            Gs = Gs.view(*x.shape[:2], *self.output_size)

        # send back to the given shape
        if reshape == 1:
            Gs = Gs.squeeze(0).squeeze(0)
        if reshape == 2:
            Gs = Gs.squeeze(0)
        return Gs


