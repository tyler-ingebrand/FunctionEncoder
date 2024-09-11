import torch


class BaseArchitecture(torch.nn.Module):

    def __init__(self):
        super(BaseArchitecture, self).__init__()

    @staticmethod
    def predict_number_params(*args, **kwargs):
        raise NotImplementedError()