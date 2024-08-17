from typing import Any, Union

from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder import BaseCallback


class TensorboardCallback(BaseCallback):
    """ Logs stats from function encoder. """
    def __init__(self,
                 logdir:Union[str, None]=None,
                 tensorboard:Union[None, SummaryWriter]=None,
                 prefix="train"):
        """ Constructor for TensorboardCallback. Either logdir  or tensorboard must be provided, but not both"""
        assert logdir is not None or tensorboard is not None, "Either logdir or tensorboard must be provided"
        assert logdir is None or tensorboard is None, "Only one of logdir or tensorboard can be provided"
        super(TensorboardCallback, self).__init__()
        if logdir is not None:
            self.tensorboard = SummaryWriter(logdir)
        else:
            self.tensorboard = tensorboard
        self.total_epochs = 0
        self.prefix = prefix

    def on_training_start(self, locals: dict):
        """ Logs parameters at the start of training. """
        function_encoder = locals["self"]
        params = function_encoder._param_string()
        for key, value in params.items():
            self.tensorboard.add_text(key, value, 0)

    def on_step(self, locals: dict):
        """ Logs losses at the end of each training step. """
        # get locals we need
        function_encoder = locals["self"]
        prediction_loss = locals["prediction_loss"]
        norm = locals["norm"]

        # log
        self.tensorboard.add_scalar(f"{self.prefix}/mean_distance_squared", prediction_loss.item(), self.total_epochs)
        self.tensorboard.add_scalar(f"{self.prefix}/gradient_norm", norm.item(), self.total_epochs)
        if function_encoder.method == "least_squares":
            norm_loss = locals["norm_loss"]
            self.tensorboard.add_scalar(f"{self.prefix}/basis_function_magnitude_loss", norm_loss.item(), self.total_epochs)
        if function_encoder.average_function is not None:
            average_function_loss = locals["average_function_loss"]
            self.tensorboard.add_scalar(f"{self.prefix}/average_function_mean_distance_squared", average_function_loss.item(), self.total_epochs)
        self.total_epochs += 1