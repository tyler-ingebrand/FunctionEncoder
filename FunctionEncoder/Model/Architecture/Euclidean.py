import torch
import warnings

from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture


class Euclidean(BaseArchitecture):
    @staticmethod
    def predict_number_params(output_size, n_basis, *args, **kwargs):
        return n_basis * output_size[0]

    def __init__(self,
                 input_size:tuple[int],
                 output_size:tuple[int],
                 n_basis:int=100,
                 pretty=True):
        super(Euclidean, self).__init__()
        assert input_size[0] == 1, "Euclidean vectors have no inputs, so we use 1 for consistency with NN"
        assert len(output_size) == 1, "Euclidean vectors have a single dimension, where the size of the dimension is the dimensionality of the space."
        warnings.warn("'Euclidean' class is designed just to visualize the algorithm, it is not meant to be used in practice. \
                      Make sure you know what you are doing before using this. If you are running the EuclideanExample, you are good to go.")
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        g = torch.rand(n_basis, output_size[0])
        if pretty:
            g[1, 0] -= 0.6  # this is to make a pretty video
            if n_basis > 2:
                g[2, :] *= -1 / 2
        self.basis = torch.nn.Parameter(g)

    def forward(self, x):
        assert x.shape[-1] == 1, "Eucldiean vectors don't really have inputs, so it should be just size one for consistency with NN"
        assert x.shape[-2] == 1, "Euclidean vectors don't have data batches, so the batch size should be one for consistency with NN"
        g = self.basis.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        return g.expand(x.shape[0], 1, self.output_size[0], self.n_basis)

