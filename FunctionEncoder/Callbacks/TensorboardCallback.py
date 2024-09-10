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

        # sometimes we restart training. Only do something if this is actually the first time.
        if self.total_epochs == 0:
            # log parameters
            params = function_encoder._param_string()
            for key, value in params.items():
                self.tensorboard.add_text(key, value, 0)




    def on_step(self, locals: dict):
        """ Logs losses at the end of each training step. """
        if "prediction_loss" in locals:
            prediction_loss = locals["prediction_loss"]
            self.tensorboard.add_scalar(f"{self.prefix}/mean_distance_squared", prediction_loss.item(), self.total_epochs)
        if "norm" in locals:
            norm = locals["norm"]
            self.tensorboard.add_scalar(f"{self.prefix}/gradient_norm", norm.item(), self.total_epochs)
        if "norm_loss" in locals:
            norm_loss = locals["norm_loss"]
            self.tensorboard.add_scalar(f"{self.prefix}/basis_function_magnitude_loss", norm_loss.item(), self.total_epochs)
        if "average_function_loss" in locals:
            average_function_loss = locals["average_function_loss"]
            self.tensorboard.add_scalar(f"{self.prefix}/average_function_mean_distance_squared", average_function_loss.item(), self.total_epochs)
        if "loss" in locals: # makes sure there is something to log.
            self.total_epochs += 1