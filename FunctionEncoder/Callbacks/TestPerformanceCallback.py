import torch

from FunctionEncoder import FunctionEncoder
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback


class TestPerformanceCallback(BaseCallback):

    def __init__(self, testing_dataset, device):
        super(TestPerformanceCallback, self).__init__()
        self.testing_dataset = testing_dataset
        self.device = device

    def on_step_begin(self, function_encoder:FunctionEncoder) -> dict:
        with torch.no_grad():
            # sample testing data
            example_xs, example_ys, xs, ys, info = self.testing_dataset.sample(device=self.device)

            # compute representation
            y_hats = function_encoder.predict_from_examples(example_xs, example_ys, xs, method="least_squares")

            # measure mse
            loss = torch.mean((ys - y_hats) ** 2).item()

            return {"test/loss": loss}

