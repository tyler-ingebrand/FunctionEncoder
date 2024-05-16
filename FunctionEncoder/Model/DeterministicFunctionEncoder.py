import torch
from tqdm import trange

from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder.Model.ModelHelpers import build_model


class DeterministicFunctionEncoder(torch.nn.Module):


    def __init__(self,
                 input_size,
                 output_size,
                 n_basis,
                 hidden_size=256,
                 n_layers=4,
                 activation:str="relu",
                 method:str="least_squares",
                 ):
        assert len(input_size) == 1, "Only 1D input supported for now"
        assert len(output_size) == 1, "Only 1D output supported for now"
        super(DeterministicFunctionEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.model = build_model(self.input_size, self.output_size, self.n_basis, hidden_size, n_layers, activation)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.method = method


    def forward(self, x):
        outs = self.model(x)
        Gs = outs.view(*x.shape[:2], *self.output_size, self.n_basis)
        return Gs

    def compute_representation(self, example_xs, example_ys, method="inner_product", **kwargs):
        assert example_xs.shape[-len(self.input_size):] == self.input_size, f"example_xs must have shape (..., {self.input_size}). Expected {self.input_size}, got {example_xs.shape[-len(self.input_size):]}"
        assert example_ys.shape[-len(self.output_size):] == self.output_size, f"example_ys must have shape (..., {self.output_size}). Expected {self.output_size}, got {example_ys.shape[-len(self.output_size):]}"
        assert example_xs.shape[:-len(self.input_size)] == example_ys.shape[:-len(self.output_size)], f"example_xs and example_ys must have the same shape except for the last {len(self.input_size)} dimensions. Expected {example_xs.shape[:-len(self.input_size)]}, got {example_ys.shape[:-len(self.output_size)]}"

        # if not in terms of functions, add a function batch dimension
        reshaped = False
        if len(example_xs.shape) - len(self.input_size) == 1:
            reshaped = True
            example_xs = example_xs.unsqueeze(0)
            example_ys = example_ys.unsqueeze(0)

        # compute representation
        if method == "inner_product":
            outs = self._compute_inner_product(example_xs, example_ys)
            GTG = None
        elif method == "least_squares":
            outs, GTG = self._compute_least_squares(example_xs, example_ys, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # reshape if necessary
        if reshaped:
            assert outs.shape[0] == 1, "Expected a single function batch dimension"
            outs = outs.squeeze(0)
        return outs, GTG

    def _compute_inner_product(self, example_xs, example_ys):
        Gs = self.forward(example_xs)
        datapoint_inner_products = torch.einsum("fdmk,fdm->fdk", Gs, example_ys) # computes element wise inner products, IE dot product
        inner_products = torch.mean(datapoint_inner_products, dim=1) # Computes the function wise inner product
        return inner_products

    def _compute_least_squares(self, example_xs, example_ys, lambd=0.1):
        # get approximations
        Gs = self.forward(example_xs)

        # compute the matrix G^TG, plus some regularization term
        GTG = torch.einsum("fdmk, fdml->fdkl", Gs, Gs) # computes element wise inner products, IE dot product, between all pairs of basis functions
        GTG = torch.mean(GTG, dim=1) # Computes the function wise inner product between all pairs of basis functions
        # eigen_values, R = torch.linalg.eigh(GTG)
        # GTG_reg = R @ R.transpose(-1, -2) # Regularize the matrix by setting it to be orthogonal
        GTG_reg = GTG + lambd * torch.eye(GTG.shape[-1], device=GTG.device)

        # compute the matrix G^TF
        GTF = torch.einsum("fdmk,fdm->fdk", Gs, example_ys) # computes element wise inner products, IE dot product
        GTF = torch.mean(GTF, dim=1).unsqueeze(-1) # Computes the function wise inner product

        # Compute (G^TG)^-1 G^TF
        representation = GTG_reg.inverse() @ GTF
        return representation.squeeze(-1), GTG

    def predict(self, xs, representations):
        Gs = self.forward(xs)
        y_hats = torch.einsum("fdmk,fk->fdm", Gs, representations)
        return y_hats

    def predict_from_examples(self, example_xs, example_ys, xs, method="inner_product", **kwargs):
        representations, _ = self.compute_representation(example_xs, example_ys, method=method, **kwargs)
        y_hats = self.predict(xs, representations)
        return y_hats

    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    logdir=None,
                    progress_bar=True,
                    callback:BaseCallback=None):
        assert dataset.data_type == "deterministic", "Only deterministic datasets are supported for DeterministicFunctionEncoder"
        # set device
        device = next(self.parameters()).device

        # if logdir is provided, use tensorboard
        if logdir is not None:
            writer = SummaryWriter(logdir)

        # method to use for representation during training
        method = self.method
        assert method in ["inner_product", "least_squares"], f"Unknown method: {method}"

        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            example_xs, example_ys, xs, ys, _ = dataset.sample(device=device)

            # approximate functions
            if method == "inner_product":
                # approximate
                y_hats_ip = self.predict_from_examples(example_xs, example_ys, xs, method="inner_product")
                prediction_loss = torch.mean((y_hats_ip - ys) ** 2)
                loss = prediction_loss
            if method == "least_squares":

                representation_ls, gram = self.compute_representation(example_xs, example_ys, method="least_squares")
                y_hats = self.predict(xs, representation_ls)
                prediction_loss = torch.mean((y_hats - ys) ** 2)
                norm_loss = ((torch.diagonal(gram, dim1=1, dim2=2) - 1)**2).mean()
                loss = prediction_loss + norm_loss

            # optimize
            self.opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.opt.step()

            # callback
            if callback is not None:
                res = callback.on_step_begin(self)
                if logdir is not None:
                    for k, v in res.items():
                        writer.add_scalar(k, v, epoch)

            # log
            if logdir is not None:
                writer.add_scalar("train/prediction_loss", loss.item(), epoch)
                writer.add_scalar("train/grad_norm", norm, epoch)
                if method == "least_squares":
                    writer.add_scalar("train/norm_loss", norm_loss.item(), epoch)