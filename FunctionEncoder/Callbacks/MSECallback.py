from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder import FunctionEncoder, BaseDataset
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback


class MSECallback(BaseCallback):

    def __init__(self,
                 testing_dataset:BaseDataset,
                 logdir: Union[str, None] = None,
                 tensorboard: Union[None, SummaryWriter] = None,
                 prefix="test",
                 ):
        """ Constructor for MSECallback. Either logdir  or tensorboard must be provided, but not both"""
        assert logdir is not None or tensorboard is not None, "Either logdir or tensorboard must be provided"
        assert logdir is None or tensorboard is None, "Only one of logdir or tensorboard can be provided"
        super(MSECallback, self).__init__()
        self.testing_dataset = testing_dataset
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
            example_xs, example_ys, query_xs, query_ys, info = self.testing_dataset.sample()

            # compute representation
            y_hats = function_encoder.predict_from_examples(example_xs, example_ys, query_xs, method="least_squares")

            # measure mse
            loss = torch.mean((query_ys - y_hats) ** 2).item()

            # log results
            self.tensorboard.add_scalar(f"{self.prefix}/mse", loss, self.total_epochs)
            self.total_epochs += 1

