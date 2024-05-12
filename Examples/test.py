
import torch

# vector valued least squares
d = 100 # num data points
m = 3 # vec size
k = 10 # num basis
lambd = 0.1

# create matrices
A = torch.rand(d,m,k)
b = torch.rand(d,m)

# calculate weights
ATA = torch.einsum("dmk, dml -> kl", A, A) + lambd * torch.eye(k)