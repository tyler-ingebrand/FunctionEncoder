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
                 negative_logit:float=-5.0,
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
        elif method == "grad_mle":
            outs = self._compute_mle(example_xs, example_ys, **kwargs)
            GTG = None
        else:
            raise ValueError(f"Unknown method: {method}")

        # reshape if necessary
        if reshaped:
            assert outs.shape[0] == 1, "Expected a single function batch dimension"
            outs = outs.squeeze(0)
        return outs, GTG

    def _compute_inner_product(self, example_xs, example_ys):

        # generate data representing distributions
        all_points, all_logits = self.convert_samples_to_logits(example_ys)
        all_logits_matrix = all_logits.unsqueeze(1) - all_logits.unsqueeze(2)

        # generate data from basis
        basis_logits = self.forward(None, all_points)
        base_logits_matrix = basis_logits.unsqueeze(1) - basis_logits.unsqueeze(2)

        # compute the inner product
        representation = torch.mean(base_logits_matrix * all_logits_matrix.unsqueeze(-1), dim=(1,2)) * 0.5 * self.volume

        return representation




    def _compute_least_squares(self, example_xs, example_ys, lambd=0.1, n_samples=None):
        # generate data representing distributions
        all_points, all_logits = self.convert_samples_to_logits(example_ys, n_samples=n_samples)
        all_logits_matrix = all_logits.unsqueeze(1) - all_logits.unsqueeze(2)

        # generate data from basis
        basis_logits = self.forward(None, all_points)
        base_logits_matrix = basis_logits.unsqueeze(1) - basis_logits.unsqueeze(2)

        base_logits_self_sim = base_logits_matrix.unsqueeze(3) * base_logits_matrix.unsqueeze(4)
        gram = torch.mean(base_logits_self_sim, dim=(1, 2)) * 0.5 * self.volume
        gram_reg = gram + lambd * torch.eye(self.n_basis, device=gram.device)

        # compute the inner product
        GF = torch.mean(base_logits_matrix * all_logits_matrix.unsqueeze(-1), dim=(1,2)) * 0.5 * self.volume
        representation = torch.einsum("fkl, fk->fl", torch.inverse(gram_reg), GF)

        return representation, gram

    def _compute_mle(self, example_xs, example_ys, grad_steps=100_000):
        # init the reps to grad
        representations = torch.randn(example_ys.shape[0], self.n_basis, device=example_ys.device)
        representations *=  0.1
        representations.requires_grad = True
        opti = torch.optim.Adam([representations], lr=1e-3)

        # collect random data to compute sums with
        # get random every time
        with torch.no_grad():
            random_ys = self.sample(example_ys.shape[0], example_ys.shape[1] * 10, example_ys.device)
            G_random_logits = self.forward(None, random_ys)

        # compute basis just once, no grad needed
        with torch.no_grad():
            G_example_logits = self.forward(None, example_ys)

        # use grad descent to find the mle estimator of the representations
        tbar = trange(grad_steps)
        for i in tbar:
            # compute the log probs via the representations
            random_logits = torch.einsum("fdk,fk->fd", G_random_logits, representations)

            # Compute sum of exponentials
            e_random_logits = torch.exp(random_logits)
            sums = torch.mean(e_random_logits, dim=1) * self.volume

            # compute example log probs
            log_prob = -example_ys.shape[1] * torch.log(sums + 1e-7) + torch.einsum("fdk,fk->f", G_example_logits, representations)


            # descent step
            loss = -torch.mean(log_prob)
            opti.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(representations, 1)
            opti.step()
            tbar.set_description(f"Loss: {loss:.3f}, Norm: {norm:.3f}")

        representations = representations.detach()  # no need to track gradients anymore
        return representations


    def predict(self, xs, ys, representations):
        Gs = self.forward(xs, ys)
        log_probs = torch.einsum("fdk,fk->fd", Gs, representations)
        return log_probs # log(p(y|x))

    def predict_from_examples(self, example_xs, example_ys, xs, ys, method="inner_product", **kwargs):
        representations, _ = self.compute_representation(example_xs, example_ys, method=method, **kwargs)
        log_probs = self.predict(xs, ys, representations)
        return log_probs # log(p(y|x))

    def convert_samples_to_logits(self, ys, n_samples=None):
        # generate data representing distributions
        true_points = ys
        true_point_logits = torch.ones(true_points.shape[0], true_points.shape[1], device=true_points.device) * self.positive_logit

        n_samples = n_samples or ys.shape[1]
        random_points = self.sample(ys.shape[0], n_samples, ys.device)
        random_point_logits = torch.ones(random_points.shape[0], random_points.shape[1], device=random_points.device) * self.negative_logit

        all_points = torch.cat([true_points, random_points], dim=1)
        all_logits = torch.cat([true_point_logits, random_point_logits], dim=1)
        return all_points, all_logits

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
                representation_ip, _ = self.compute_representation(example_xs, example_ys, method="inner_product")
                all_points, all_logits = self.convert_samples_to_logits(ys)
                approximate_logits = self.predict(None, all_points, representation_ip)
                error_vector = all_logits - approximate_logits
                error_matrix = error_vector.unsqueeze(1) - error_vector.unsqueeze(2)
                error_ip = torch.mean(error_matrix ** 2, dim=(1, 2))
                logit_loss = torch.mean(error_ip)
                loss = logit_loss
            if method == "least_squares":
                representation_ls, gram = self.compute_representation(example_xs, example_ys, method="least_squares")

                # logit loss
                all_points, all_logits = self.convert_samples_to_logits(ys)
                approximate_logits = self.predict(None, all_points, representation_ls)
                error_vector = all_logits - approximate_logits
                error_matrix = error_vector.unsqueeze(1) - error_vector.unsqueeze(2)
                error_ip = torch.mean(error_matrix ** 2, dim=(1, 2))
                logit_loss = torch.mean(error_ip)

                if epoch == epochs - 1:
                    pass
                # gram loss
                norm_loss = ((torch.diagonal(gram, dim1=-2, dim2=-1) - 1)** 2).mean()

                loss = logit_loss + norm_loss

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