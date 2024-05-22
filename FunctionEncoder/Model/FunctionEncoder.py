import torch
from tqdm import trange

from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder.Model.ModelHelpers import build_model


class FunctionEncoder(torch.nn.Module):


    def __init__(self,
                 input_size,
                 output_size,
                 data_type,
                 n_basis=100,
                 hidden_size=256,
                 n_layers=4,
                 activation:str="relu",
                 method:str="least_squares",
                 ):
        assert len(input_size) == 1, "Only 1D input supported for now"
        assert input_size[0] >= 1, "Input size must be at least 1"
        assert len(output_size) == 1, "Only 1D output supported for now"
        assert output_size[0] >= 1, "Output size must be at least 1"
        assert data_type in ["deterministic", "stochastic"], f"Unknown data type: {data_type}"
        super(FunctionEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.model = build_model(self.input_size, self.output_size, self.n_basis, hidden_size, n_layers, activation)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.method = method
        self.data_type = data_type


    def forward(self, x):
        assert x.shape[-1] == self.input_size[0], f"Expected input size {self.input_size[0]}, got {x.shape[-1]}"
        reshape = None
        if len(x.shape) == 1:
            reshape = 1
            x = x.reshape(1, 1, -1)
        if len(x.shape) == 2:
            reshape = 2
            x = x.unsqueeze(0)

        # this is the main part of this function. The rest is just error handling
        outs = self.model(x)
        Gs = outs.view(*x.shape[:2], *self.output_size, self.n_basis)

        # send back to the given shape
        if reshape == 1:
            Gs = Gs.squeeze(0).squeeze(0)
        if reshape == 2:
            Gs = Gs.squeeze(0)
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
            representation = self._compute_inner_product_representation(example_xs, example_ys)
            gram = None
        elif method == "least_squares":
            representation, gram = self._compute_least_squares_representation(example_xs, example_ys, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # reshape if necessary
        if reshaped:
            assert representation.shape[0] == 1, "Expected a single function batch dimension"
            representation = representation.squeeze(0)
        return representation, gram

    def _deterministic_inner_product(self, fs, gs):
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _stochastic_inner_product(self, fs, gs):
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

        # fs should be fdn, gs should be fdnk
        # compute means
        mean_f = torch.mean(fs, dim=1, keepdim=True)
        mean_g = torch.mean(gs, dim=1, keepdim=True)
        fs = fs - mean_f
        gs = gs - mean_g

        # compute inner products
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # reshape
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _inner_product(self, fs, gs):
        if self.data_type == "deterministic":
            return self._deterministic_inner_product(fs, gs)
        elif self.data_type == "stochastic":
            return self._stochastic_inner_product(fs, gs)
        else:
            raise ValueError(f"Unknown data type: {self.data_type}")

    def _compute_inner_product_representation(self, example_xs, example_ys):
        Gs = self.forward(example_xs)
        inner_products = self._inner_product(Gs, example_ys)
        return inner_products

    def _compute_least_squares_representation(self, example_xs, example_ys, lambd=0.1):
        # get approximations
        Gs = self.forward(example_xs)

        # compute gram
        gram = self._inner_product(Gs, Gs)
        gram_reg = gram + lambd * torch.eye(self.n_basis, device=gram.device)

        # compute the matrix G^TF
        ip_representation = self._inner_product(Gs, example_ys)

        # Compute (G^TG)^-1 G^TF
        ls_representation = torch.einsum("fkl,fl->fk", gram_reg.inverse(), ip_representation) # this is just batch matrix multiplication
        return ls_representation, gram

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
        # set device
        device = next(self.parameters()).device

        # if logdir is provided, use tensorboard
        if logdir is not None:
            writer = SummaryWriter(logdir)

        # method to use for representation during training
        assert self.method in ["inner_product", "least_squares"], f"Unknown method: {self.method}"

        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            example_xs, example_ys, xs, ys, _ = dataset.sample(device=device)

            # approximate functions, compute error
            representation, gram = self.compute_representation(example_xs, example_ys, method=self.method)
            y_hats = self.predict(xs, representation)
            error_vector = y_hats - ys
            prediction_loss = self._inner_product(error_vector, error_vector).mean()

            # LS requires regularization since it does not care about the scale of basis
            # so we force basis to move towards unit norm. They dont actually need to be unit, but this prevents them
            # from going to infinity.
            if self.method == "least_squares":
                norm_loss = ((torch.diagonal(gram, dim1=1, dim2=2) - 1)**2).mean()

            # optimize
            loss = prediction_loss + norm_loss if self.method == "least_squares" else prediction_loss
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
                if self.method == "least_squares":
                    writer.add_scalar("train/norm_loss", norm_loss.item(), epoch)