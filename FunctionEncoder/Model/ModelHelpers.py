import torch



# Builds a function encoder with the desired specifications
def build_model( input_size, output_size, n_basis, hidden_size, n_layers, activation):
    assert type(input_size) == tuple, "input_size must be a tuple"
    assert type(output_size) == tuple, "output_size must be a tuple"

    # get inputs
    input_size = input_size[0]  # only 1D input supported for now
    output_size = output_size[0] * n_basis

    # build net
    layers = []
    layers.append(torch.nn.Linear(input_size, hidden_size))
    layers.append(get_activation(activation))
    for _ in range(n_layers - 2):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        layers.append(get_activation(activation))
    layers.append(torch.nn.Linear(hidden_size, output_size))
    return torch.nn.Sequential(*layers)


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