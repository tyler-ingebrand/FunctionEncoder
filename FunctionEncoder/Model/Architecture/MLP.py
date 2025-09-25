import torch

from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture


# Returns the desired activation function by name
def get_activation( activation):
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation: {activation}")

class MLP(BaseArchitecture):

    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size=256, n_layers=4, average_function=False, *args, **kwargs):
        input_size = input_size[0]
        output_size = output_size[0] * n_basis if not average_function else output_size[0]
        n_params =  input_size * hidden_size + hidden_size + \
                    (n_layers - 2) * hidden_size * hidden_size + (n_layers - 2) * hidden_size + \
                    hidden_size * output_size + output_size
        return n_params

    def __init__(self,
                 input_size:tuple[int],
                 output_size:tuple[int],
                 n_basis:int,
                 average_function:bool=False,
                 hidden_size:int=256,
                 n_layers:int=4,
                 activation:str="relu"
                 ):
        assert len(input_size) == 1, "MLP only supports 1D input"
        assert len(output_size) == 1, "MLP only supports 1D outputs"
        super(MLP, self).__init__(
            input_size=input_size,
            output_size=output_size,
            n_basis=n_basis,
            average_function=average_function,
        )
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.average_function = average_function


        # get inputs
        input_size = input_size[0]  # only 1D input supported for now
        output_size = output_size[0] * n_basis if not average_function else output_size[0]

        # build net
        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_size))
        layers.append(get_activation(activation))
        for _ in range(n_layers - 2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(get_activation(activation))
        layers.append(torch.nn.Linear(hidden_size, output_size))
        self.model = torch.nn.Sequential(*layers)

        # verify number of parameters
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = self.predict_number_params(self.input_size, self.output_size, n_basis, hidden_size, n_layers, average_function=average_function)
        assert n_params == estimated_n_params, f"Model has {n_params} parameters, but expected {estimated_n_params} parameters."


    def forward(self, x):
        assert x.shape[-1] == self.input_size[0], f"Expected input size {self.input_size[0]}, got {x.shape[-1]}"

        # forward pass
        outs = self.model(x)

        # if learning a basis, reshape into correct format
        if not self.average_function:
            outs = outs.view(*x.shape[:-len(self.input_size)], *self.output_size, self.n_basis)

        return outs


