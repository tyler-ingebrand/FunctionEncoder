import torch


class BaseArchitecture(torch.nn.Module):

    def __init__(self):
        super(BaseArchitecture, self).__init__()

    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size, n_layers):
        raise NotImplementedError()