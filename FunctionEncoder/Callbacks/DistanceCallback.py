from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder import FunctionEncoder, BaseDataset
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback


class DistanceCallback(BaseCallback):

    def __init__(self,
                 testing_dataset:BaseDataset,
                 device:Union[str, torch.device],
                 logdir: Union[str, None] = None,
                 tensorboard: Union[None, SummaryWriter] = None,
                 prefix="test",
                 ):
        """ Constructor for MSECallback. Either logdir  or tensorboard must be provided, but not both"""
        assert logdir is not None or tensorboard is not None, "Either logdir or tensorboard must be provided"
        assert logdir is None or tensorboard is None, "Only one of logdir or tensorboard can be provided"
        super(DistanceCallback, self).__init__()
        self.testing_dataset = testing_dataset
        self.device = device
        if logdir is not None:
            self.tensorboard = SummaryWriter(logdir)
        else:
            self.tensorboard = tensorboard
        self.prefix = prefix
        self.total_epochs = 0

    def on_step(self, locals:dict):
        with torch.no_grad():
            function_encoder = locals["self"]

            # sample testing data
            example_xs, example_ys, xs, ys, info = self.testing_dataset.sample()

            # compute representation
            y_hats = function_encoder.predict_from_examples(example_xs, example_ys, xs, method="least_squares")

            # measure mse
            loss = function_encoder._distance(y_hats, ys, squared=True).mean()

            # log results
            self.tensorboard.add_scalar(f"{self.prefix}/distance", loss, self.total_epochs)
            self.total_epochs += 1

