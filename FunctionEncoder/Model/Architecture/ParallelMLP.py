import math

import torch

from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.MLP import get_activation


class ParallelLinear(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_parallel, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        n = n_inputs
        m = n_outputs
        p = n_parallel
        self.W = torch.nn.Parameter(torch.zeros((p, m, n), **factory_kwargs))
        self.b = torch.nn.Parameter(torch.zeros((p, m), **factory_kwargs))
        self.n = n
        self.m = m
        self.p = p
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # this function was designed for a single layer, so we need to do it k times to not break their code.
        # this is slow but we only pay this cost once.
        for i in range(self.p):
            torch.nn.init.kaiming_uniform_(self.W[i, :, :], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W[i, :, :])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.b[i, :], -bound, bound)

    def forward(self, x):
        assert x.shape[-2] == self.n, f"Input size of model '{self.n}' does not match input size of data '{x.shape[-2]}'"
        assert x.shape[-1] == self.p, f"Batch size of model '{self.p}' does not match batch size of data '{x.shape[-1]}'"
        y = torch.einsum("pmn,...np->...mp", self.W, x) + self.b.T.reshape(1, 1, self.m, self.p)
        return y

    def num_params(self):
        return self.W.numel() + self.b.numel()

    def __repr__(self):
        return f"ParallelLinear({self.n}, {self.m}, {self.p})"

    def __str__(self):
        return self.__repr__()
    def __call__(self, x):
        return self.forward(x)



class ParallelMLP(BaseArchitecture):
    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size, n_layers, *args, **kwargs):
        input_size = input_size[0]
        output_size = output_size[0]
        # +1 accounts for bias
        n_params =  (input_size+1) * hidden_size + \
                    (hidden_size+1) * hidden_size * (n_layers - 2) + \
                    (hidden_size+1) * output_size
        n_params *= n_basis
        return n_params


    def __init__(self,
                 input_size:tuple[int],
                 output_size:tuple[int],
                 n_basis:int=100,
                 hidden_size:int=77,
                 n_layers:int=4,
                 activation:str="relu"):
        super(ParallelMLP, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis


        # get inputs
        input_size = input_size[0]  # only 1D input supported for now
        output_size = output_size[0]

        # build net
        layers = []
        layers.append(ParallelLinear(input_size, hidden_size, n_basis))
        layers.append(get_activation(activation))
        for _ in range(n_layers - 2):
            layers.append(ParallelLinear(hidden_size, hidden_size, n_basis))
            layers.append(get_activation(activation))
        layers.append(ParallelLinear(hidden_size, output_size, n_basis))
        self.model = torch.nn.Sequential(*layers)

        # verify number of parameters
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = self.predict_number_params(self.input_size, self.output_size, n_basis, hidden_size, n_layers)
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

        # need to append the batch dimension as the last dim
        x = x.unsqueeze(-1).repeat(1, 1, 1, self.n_basis)

        # this is the main part of this function. The rest is just error handling
        outs = self.model(x)
        if self.n_basis > 1:
            Gs = outs.view(*x.shape[:2], *self.output_size, self.n_basis)
        else:
            Gs = outs.view(*x.shape[:2], *self.output_size)

        # send back to the given shape
        if reshape == 1:
            Gs = Gs.squeeze(0).squeeze(0)
        if reshape == 2:
            Gs = Gs.squeeze(0)
        return Gs


