from typing import Callable

import torch
from tqdm import trange

from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder.Model.ModelHelpers import build_model


class StochasticFunctionEncoder(torch.nn.Module):


    def __init__(self,
                 input_size:tuple,
                 output_size:tuple,
                 n_basis:int,
                 sample:Callable, # samples some number of points from the input space
                 volume:float,
                 hidden_size=256,
                 n_layers=4,
                 activation:str="relu",
                 method:str="inner_product",
                 positive_logit:float=5.0,
                 negative_logit:float=0.0,
                 ):
        assert len(input_size) == 1, "Only 1D input supported for now"
        assert len(output_size) == 1, "Only 1D output supported for now"
        super(StochasticFunctionEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        inputs = (self.input_size[0] + self.output_size[0], )
        outputs = (1,)
        self.model = build_model(inputs, outputs, self.n_basis, hidden_size, n_layers, activation)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.method = method
        self.positive_logit = positive_logit
        self.negative_logit = negative_logit
        self.sample = sample
        self.volume = volume


    def forward(self, x, y):
        if x is not None:
            ins = torch.cat([x, y], dim=-1)
        else:
            ins = y
        outs = self.model(ins)
        # Gs = outs.view(*y.shape[:2], self.n_basis)
        return outs

    def compute_representation(self, example_xs, example_ys, method="inner_product", **kwargs):
        assert example_xs is None or example_xs.shape[-len(self.input_size):] == self.input_size, f"example_xs must have shape (..., {self.input_size}). Expected {self.input_size}, got {example_xs.shape[-len(self.input_size):]}"
        assert example_ys.shape[-len(self.output_size):] == self.output_size, f"example_ys must have shape (..., {self.output_size}). Expected {self.output_size}, got {example_ys.shape[-len(self.output_size):]}"
        assert example_xs is None or example_xs.shape[:-len(self.input_size)] == example_ys.shape[:-len(self.output_size)], f"example_xs and example_ys must have the same shape except for the last {len(self.input_size)} dimensions. Expected {example_xs.shape[:-len(self.input_size)]}, got {example_ys.shape[:-len(self.output_size)]}"

        # if not in terms of functions, add a function batch dimension
        reshaped = False
        if len(example_ys.shape) - len(self.input_size) == 1:
            reshaped = True
            if example_xs is not None:
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

        # generate data representing distributions
        true_points = example_ys
        true_point_logits = torch.ones(true_points.shape[0], true_points.shape[1], device=true_points.device) * self.positive_logit

        random_points = self.sample(example_ys.shape[0], example_ys.shape[1], example_ys.device)
        random_point_logits = torch.ones(random_points.shape[0], random_points.shape[1], device=random_points.device) * self.negative_logit

        all_points = torch.cat([true_points, random_points], dim=1)
        all_logits = torch.cat([true_point_logits, random_point_logits], dim=1)
        all_logits_matrix = all_logits.unsqueeze(1) - all_logits.unsqueeze(2)

        # generate data from basis
        basis_logits = self.forward(None, all_points)
        base_logits_matrix = basis_logits.unsqueeze(1) - basis_logits.unsqueeze(2)

        # compute the inner product
        representation = torch.einsum("fdek,fde->fk", base_logits_matrix, all_logits_matrix) * 0.5 * base_logits_matrix.shape[1] * self.volume

        return representation









    def _compute_least_squares(self, example_xs, example_ys, lambd=0.1):
        raise Exception("Not implemented yet")
        # get approximations
        Gs = self.forward(example_xs, example_ys)

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

    def predict(self, xs, ys, representations):
        Gs = self.forward(xs, ys)
        log_probs = torch.einsum("fdk,fk->fd", Gs, representations)
        return log_probs # log(p(y|x))

    def predict_from_examples(self, example_xs, example_ys, xs, ys, method="inner_product", **kwargs):
        representations, _ = self.compute_representation(example_xs, example_ys, method=method, **kwargs)
        log_probs = self.predict(xs, ys, representations)
        return log_probs # log(p(y|x))

    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    logdir=None,
                    progress_bar=True,
                    callback:BaseCallback=None):
        assert dataset.data_type == "stochastic", "Only deterministic datasets are supported for StohchasticFunctionEncoder"
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
                logits_ip = self.predict_from_examples(example_xs, example_ys, xs, ys, method="inner_product")
                logit_loss = -torch.mean(logits_ip)
                loss = logit_loss
            if method == "least_squares":
                raise Exception("Not implemented yet")
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
                writer.add_scalar("train/logit_loss", logit_loss.item(), epoch)
                writer.add_scalar("train/grad_norm", norm, epoch)
                if method == "least_squares":
                    writer.add_scalar("train/norm_loss", norm_loss.item(), epoch)