from typing import Iterable, List

import torch

from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.MLP import get_activation, MLP


class ConvLayers(BaseArchitecture):
    @staticmethod
    def predict_flatten_size(input_size:tuple[int],
                             conv_kernel_size:int=3,
                             n_channels:List[int]=None,
                             maxpool_kernel_size: int = 2,
                             maxpool_stride:int = 2,
                             *args, **kwargs):
        input_channels, height, width = input_size
        output_channels = n_channels[-1]
        maxpool_squeeze_size = maxpool_stride**(len(n_channels) - 1)
        flatten_size = output_channels * (height//maxpool_squeeze_size) * (width//maxpool_squeeze_size)
        return flatten_size

    @staticmethod
    def predict_number_params(input_size:tuple[int],
                             conv_kernel_size:int=3,
                             n_channels:List[int]=None,
                             maxpool_kernel_size: int = 2,
                             maxpool_stride:int = 2,
                             *args, **kwargs):
        n_params = 0
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            n_params += (conv_kernel_size*conv_kernel_size)*(in_channels)*(out_channels) + out_channels
        return n_params
    def __init__(self,
                 input_size:tuple[int],
                 conv_kernel_size:int=3,
                 n_channels:List[int]=None,
                 maxpool_kernel_size: int = 2,
                 maxpool_stride:int = 2,
                 ):
        super(ConvLayers, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert len(input_size) == 3, "input_size must be a tuple of length 3, CHW"
        assert input_size[-3] <= 4, f"input_size[-3] must be <= 4 for RGB (and maybe D). Got {input_size[-3]}. Image order should be CHW. "
        assert conv_kernel_size > 0, "kernel_size must be > 0"
        assert n_channels is None or (len(n_channels) > 0 and all([n > 0 for n in n_channels]) and n_channels[0]==input_size[0]), "n_channels must be a list of integers > 0"
        assert maxpool_kernel_size > 0, "maxpool_size must be > 0"
        assert maxpool_stride > 0, "maxpool_stride must be > 0"
        if n_channels is None:
            n_channels = [input_size[0], 2*input_size[0], 4*input_size[0], 8*input_size[0]]

        self.input_size = input_size
        self.conv_kernel_size = conv_kernel_size
        self.n_channels = n_channels
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride

        # A bunch of conv layers
        layers = []
        for in_channels, out_channels in zip(n_channels[:-1], n_channels[1:]):
            layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, padding=1))
            layers.append(torch.nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_kernel_size))
        layers.append(torch.nn.Flatten())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        assert len(x.shape) in [3,4], f"Expected 3 or 4 dimensions, got {len(x.shape)} with values {x.shape}"
        assert x.shape[-1] == self.input_size[-1], f"Expected {self.input_size[-1]} width, got {x.shape[-1]}"
        assert x.shape[-2] == self.input_size[-2], f"Expected {self.input_size[-2]} height, got {x.shape[-2]}"
        assert x.shape[-3] == self.input_size[-3], f"Expected {self.input_size[-3]} channels, got {x.shape[-3]}"
        return self.model(x)



class CNN(BaseArchitecture):

    @staticmethod
    def predict_number_params(input_size:tuple[int],
                             output_size:tuple[int],
                             n_basis:int=100,
                             hidden_size:int=256,
                             n_layers:int=3,
                             activation:str="relu",
                             conv_kernel_size: int = 3,
                             n_channels: List[int] = None,
                             maxpool_kernel_size: int = 2,
                             maxpool_stride: int = 2,):

        flatten_size = ConvLayers.predict_flatten_size(input_size, conv_kernel_size, n_channels, maxpool_kernel_size, maxpool_stride)
        n_params_conv = ConvLayers.predict_number_params(input_size, conv_kernel_size, n_channels, maxpool_kernel_size, maxpool_stride)
        n_params_mlp = MLP.predict_number_params((flatten_size,), output_size, n_basis, hidden_size, n_layers)
        return n_params_conv + n_params_mlp

    def __init__(self,
                 input_size:tuple[int],
                 output_size:tuple[int],
                 n_basis:int=100,
                 hidden_size:int=256,
                 n_layers:int=3,
                 activation:str="relu",
                 conv_kernel_size: int = 3,
                 n_channels: List[int] = None,
                 maxpool_kernel_size: int = 2,
                 maxpool_stride: int = 2,
                 ):
        super(CNN, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        assert len(input_size) == 3, "input_size must be a tuple of length 3, CHW"
        assert input_size[-3] <= 4, f"input_size[-3] must be <= 4 for RGB (and maybe D). Got {input_size[-3]}. Image order should be CHW. "
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert len(input_size) == 3, "input_size must be a tuple of length 3, CHW"
        assert input_size[-3] <= 4, f"input_size[-3] must be <= 4 for RGB (and maybe D). Got {input_size[-3]}. Image order should be CHW. "
        assert conv_kernel_size > 0, "kernel_size must be > 0"
        assert n_channels is None or (len(n_channels) > 0 and all([n > 0 for n in n_channels]) and n_channels[0] == input_size[0]), "n_channels must be a list of integers > 0"
        assert maxpool_kernel_size > 0, "maxpool_size must be > 0"
        assert maxpool_stride > 0, "maxpool_stride must be > 0"
        if n_channels is None:
            n_channels = [input_size[0], 2 * input_size[0], 4 * input_size[0], 8 * input_size[0]]

        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis

        # build net
        flatten_size = ConvLayers.predict_flatten_size(input_size, conv_kernel_size, n_channels, maxpool_kernel_size, maxpool_stride)
        layers = []
        layers.append(ConvLayers(input_size, conv_kernel_size, n_channels, maxpool_kernel_size, maxpool_stride))
        layers.append(MLP((flatten_size,), output_size, n_basis, hidden_size, n_layers, activation))
        self.model = torch.nn.Sequential(*layers)

        # verify correct number of parameters
        n_params = sum([p.numel() for p in self.parameters()])
        expected_params = self.predict_number_params(input_size, output_size, n_basis, hidden_size, n_layers, activation, conv_kernel_size, n_channels, maxpool_kernel_size, maxpool_stride)
        assert n_params == expected_params, f"Expected {expected_params} parameters, got {n_params}"


    def forward(self, x):
        assert len(x.shape) >= 3, f"Expected at least 3 dimensions, got {len(x.shape)} with values {x.shape}"
        assert x.shape[-1] == self.input_size[-1], f"Expected {self.input_size[-1]} channels, got {x.shape[-1]}"
        assert x.shape[-2] == self.input_size[-2], f"Expected {self.input_size[-2]} height, got {x.shape[-2]}"
        assert x.shape[-3] == self.input_size[-3], f"Expected {self.input_size[-3]} width, got {x.shape[-3]}"


        reshape = None
        if len(x.shape) == 3:
            reshape = 1
            x = x.reshape(1, 1, *x.shape)
        if len(x.shape) == 4:
            reshape = 2
            x = x.reshape(1, *x.shape)

        # this is the main part of this function. The rest is just error handling
        # flatten the batch dims. Torch only supports 1 batch dim for images
        outs = self.model(x.reshape(-1, *x.shape[2:]))
        outs = outs.reshape(x.shape[0], x.shape[1], *outs.shape[1:])

        # reshape output dims
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


