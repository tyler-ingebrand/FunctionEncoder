import torch

from FunctionEncoder import FunctionEncoder

# size of the input and output spaces
num_inputs, num_outputs = 9, 2

# number of functions to sample. Basically a batch dimension
n_functions = 10
# number of data points to sample PER function
n_datapoints = 1000

# init basis functions
basis_functions = [torch.nn.Sequential(
    torch.nn.Linear(num_inputs, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, num_outputs),
) for _ in range(100)
]


# initialize an untrained model
model = FunctionEncoder(basis_functions)

# sample example xs, example ys, and query xs
example_xs = torch.rand(n_functions, n_datapoints, num_inputs)
example_ys = torch.rand(n_functions, n_datapoints, num_outputs)
query_xs = torch.rand(n_functions, n_datapoints, num_inputs)

# do a forward pass of the model. Does the following:
# 1) uses least squares to compute the coefficients of the basis functions from the example data
# 2) uses the coefficients to compute the output of the model for the query data
query_y_hats, gram = model(example_xs, example_ys, query_xs)
# this forward pass is differentiable, so you can use it in loss functions for back prop.

print("example xs shape:", example_xs.shape)
print("example ys shape:", example_ys.shape)
print("query xs shape:", query_xs.shape)
print("query y hats shape:", query_y_hats.shape)

# you can also fetch the coefficients and the gram matrix
coeffs, gram_matrix = model.compute_representation(example_xs, example_ys)
print("coeffs shape:", coeffs.shape)
print("gram matrix shape:", gram_matrix.shape)