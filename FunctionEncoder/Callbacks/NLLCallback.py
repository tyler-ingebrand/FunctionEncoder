from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder import FunctionEncoder, BaseDataset
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback


class NLLCallback(BaseCallback):

    def __init__(self,
                 testing_dataset:BaseDataset,
                 logdir: Union[str, None] = None,
                 tensorboard: Union[None, SummaryWriter] = None,
                 prefix:str="test",
                 ):
        super(NLLCallback, self).__init__()
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
            example_xs, example_ys, xs, ys, info = self.testing_dataset.sample()

            # compute representation
            logits = function_encoder.predict_from_examples(example_xs, example_ys, xs, method=function_encoder.method)

            # measure mean_log_prob
            loss = -torch.mean(logits)

            # log results
            self.tensorboard.add_scalar(f"{self.prefix}/nll", loss, self.total_epochs)
            self.total_epochs += 1

