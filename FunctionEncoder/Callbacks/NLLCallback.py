import torch

from FunctionEncoder import FunctionEncoder
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback


class NLLCallback(BaseCallback):

    def __init__(self, testing_dataset, device):
        super(NLLCallback, self).__init__()
        self.testing_dataset = testing_dataset
        self.device = device

    def on_step(self, function_encoder:FunctionEncoder) -> dict:
        with torch.no_grad():
            # sample testing data
            example_xs, example_ys, xs, ys, info = self.testing_dataset.sample(device=self.device)

            # compute representation
            logits = function_encoder.predict_from_examples(example_xs, example_ys, xs, method=function_encoder.method)

            # measure mean_log_prob
            loss = -torch.mean(logits)

            return {"test/nll": loss}

