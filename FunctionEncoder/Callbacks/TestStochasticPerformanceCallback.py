import torch

from FunctionEncoder import StochasticFunctionEncoder
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback


class TestStochasticPerformanceCallback(BaseCallback):

    def __init__(self, testing_dataset, device):
        super(TestStochasticPerformanceCallback, self).__init__()
        self.testing_dataset = testing_dataset
        self.device = device

    def on_step_begin(self, function_encoder:StochasticFunctionEncoder) -> dict:
        with torch.no_grad():
            # sample testing data
            example_xs, example_ys, xs, ys, info = self.testing_dataset.sample(device=self.device)

            # compute representation
            logits = function_encoder.predict_from_examples(example_xs, example_ys, xs, ys, method="mle")

            # measure mean_log_prob
            loss = -torch.mean(logits)

            return {"test/neg_log_likelihood": loss}

