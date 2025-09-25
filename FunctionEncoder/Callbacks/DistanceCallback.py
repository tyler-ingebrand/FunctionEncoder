from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder import FunctionEncoder, BaseDataset
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from FunctionEncoder.Model.FunctionEncoder import recursive_to_device


class DistanceCallback(BaseCallback):

    def __init__(self,
                 dataloader:torch.utils.data.DataLoader,
                 logdir: Union[str, None] = None,
                 tensorboard: Union[None, SummaryWriter] = None,
                 prefix="test",
                 ):
        """ Constructor for MSECallback. Either logdir  or tensorboard must be provided, but not both"""
        assert logdir is not None or tensorboard is not None, "Either logdir or tensorboard must be provided"
        assert logdir is None or tensorboard is None, "Only one of logdir or tensorboard can be provided"
        super(DistanceCallback, self).__init__()
        self.dataloader = dataloader
        if logdir is not None:
            self.tensorboard = SummaryWriter(logdir)
        else:
            self.tensorboard = tensorboard
        self.prefix = prefix
        self.total_epochs = 0
        self.data_iter = iter(self.dataloader)

    def on_training_start(self, locals: dict) -> None:
        if self.total_epochs == 0: # logs loss before any updates.
            self.on_step(locals)

    def on_step(self, locals:dict):
        with torch.no_grad():
            function_encoder = locals["self"]

            # sample testing data
            try:
                data = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataloader)
                data = next(self.data_iter)
            example_xs, example_ys, query_xs, query_ys, info = recursive_to_device(data, function_encoder.device)


            # compute representation
            y_hats = function_encoder.predict_from_examples(example_xs, example_ys, query_xs)

            # measure mse
            loss = function_encoder._distance(y_hats, query_ys, squared=True).mean()

            # log results
            self.tensorboard.add_scalar(f"{self.prefix}/mean_distance_squared", loss, self.total_epochs)
            self.total_epochs += 1

