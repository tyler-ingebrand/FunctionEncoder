import torch
from tqdm import trange

from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder.Model.ModelHelpers import build_model


class FunctionEncoder(torch.nn.Module):


    def __init__(self,
                 input_size,
                 output_size,
                 n_basis,
                 hidden_size=256,
                 n_layers=4,
                 activation:str="relu",
                 ):
        assert len(input_size) == 1, "Only 1D input supported for now"
        assert len(output_size) == 1, "Only 1D output supported for now"
        super(FunctionEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.model = build_model(self.input_size, self.output_size, self.n_basis, hidden_size, n_layers, activation)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)



    def forward(self, x):
        outs = self.model(x)
        Gs = outs.view(*x.shape, self.n_basis)
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
        elif method == "least_squares":
            outs = self._compute_least_squares(example_xs, example_ys, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # reshape if necessary
        if reshaped:
            assert outs.shape[0] == 1, "Expected a single function batch dimension"
            outs = outs.squeeze(0)
        return outs

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
        GTG_reg = GTG + lambd * torch.eye(GTG.shape[-1], device=GTG.device)

        # compute the matrix G^TF
        GTF = torch.einsum("fdmk,fdm->fdk", Gs, example_ys) # computes element wise inner products, IE dot product
        GTF = torch.mean(GTF, dim=1).unsqueeze(-1) # Computes the function wise inner product

        # Compute (G^TG)^-1 G^TF
        representation = GTG_reg.inverse() @ GTF
        return representation.squeeze(-1)

    def predict(self, xs, representations):
        Gs = self.forward(xs)
        y_hats = torch.einsum("fdmk,fk->fdm", Gs, representations)
        return y_hats

    def predict_from_examples(self, example_xs, example_ys, xs, method="inner_product", **kwargs):
        representations = self.compute_representation(example_xs, example_ys, method=method, **kwargs)
        y_hats = self.predict(xs, representations)
        return y_hats

    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    logdir=None,
                    progress_bar=True):
        # set device
        device = next(self.parameters()).device

        # if logdir is provided, use tensorboard
        if logdir is not None:
            writer = SummaryWriter(logdir)

        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            example_xs, example_ys, xs, ys, _ = dataset.sample(device=device)

            # compute representation
            representations = self.compute_representation(example_xs, example_ys, method="inner_product")
            assert representations.shape == (example_xs.shape[0], self.n_basis), f"Expected representations to have shape ({example_xs.shape[0]}, {self.n_basis}), got {representations.shape}"

            # approximate functions
            y_hats = self.predict(xs, representations)
            assert y_hats.shape == ys.shape, f"Expected y_hats to have shape {ys.shape}, got {y_hats.shape}"

            # compute loss
            loss = torch.mean((y_hats - ys) ** 2)

            # optimize
            self.opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.opt.step()

            # log
            if logdir is not None:
                writer.add_scalar("train/loss", loss.item(), epoch)
                writer.add_scalar("train/grad_norm", norm, epoch)