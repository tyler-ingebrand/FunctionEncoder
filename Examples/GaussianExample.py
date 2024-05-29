import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import FunctionEncoder, NLLCallback, GaussianDataset, TensorboardCallback, ListCallback

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
args = parser.parse_args()

# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda"
train_method = args.train_method
seed = args.seed
load_path = args.load_path
residuals = args.residuals
if load_path is None:
    logdir = f"logs/gaussian_example/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

# create dataset
dataset = GaussianDataset()

if load_path is None:
    # create the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            method=train_method,
                            use_residuals_method=residuals).to(device)

    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = NLLCallback(dataset, device=device, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            n_basis=n_basis, 
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))



# plot
with torch.no_grad():
    n_cols, n_rows = 3, 3
    fig = plt.figure(figsize=(n_cols * 4 + 1, n_rows * 3.8))
    gs = plt.GridSpec(n_rows, n_cols + 1,  width_ratios=[4, 4, 4, 1])
    axes = [fig.add_subplot(gs[i // n_cols, i % n_cols], aspect='equal') for i in range(n_cols * n_rows)]

    example_xs, example_ys, _, _, info = dataset.sample("cuda")

    # compute pdf over full space
    # compute pdf at grid points and plot using plt
    grid = torch.arange(-1, 1, 0.02, device=device)
    xs = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1).reshape(-1, 2).expand(10, -1, -1)

    logits = model.predict_from_examples(example_xs, example_ys, xs, method=args.train_method)
    # e_logits = torch.exp(logits)
    # sums = torch.mean(e_logits, dim=1, keepdim=True) * dataset.volume
    # pdf = e_logits / sums
    # grid = grid.to("cpu").numpy()
    # pdf = pdf.to("cpu").numpy()
    # pdf = pdf.reshape(10, len(grid), len(grid))

    pdf = logits.to("cpu").numpy()
    pdf = pdf.reshape(10, len(grid), len(grid))
    grid = grid.to("cpu").numpy()



    std_devs = info["std_devs"]
    for i in range(9):
        ax = axes[i ]
        ax.contourf(grid, grid, pdf[i], levels=100, cmap="Reds", )
        ax.scatter(example_xs[i, :example_xs.shape[1]//2, 0].cpu(), example_xs[i, :example_xs.shape[1]//2, 1].cpu(), color="black", s=1, alpha=0.5)
        # ax.scatter(example_xs[i, example_xs.shape[1]//2:, 0].cpu(), example_xs[i, example_xs.shape[1]//2:, 1].cpu(), color="blue", s=1, alpha=0.5)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    # color bar
    cax = fig.add_subplot(gs[:, -1])
    cbar = plt.colorbar(ax.collections[0], cax=cax, orientation="vertical", fraction=0.1)

    plt.tight_layout()
    plt.savefig(f"{logdir}/gaussians.png")
