import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import GaussianDonutDataset
from FunctionEncoder import StochasticFunctionEncoder
from FunctionEncoder import TestStochasticPerformanceCallback

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="inner_product")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda"
train_method = args.train_method
seed = args.seed
load_path = args.load_path
if load_path is None:
    logdir = f"logs/donut_example/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)


# create dataset
dataset = GaussianDonutDataset()
sample = dataset.sample_inputs
volume = dataset.volume

if load_path is None:
    # create the model
    model = StochasticFunctionEncoder(input_size=(0,), output_size=(2,), n_basis=n_basis, method=train_method, sample=sample, volume=volume).to(device)

    # create a testing callback
    callback = TestStochasticPerformanceCallback(dataset, device=device)

    # train the model
    model.train_model(dataset, epochs=epochs, logdir=logdir, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model = StochasticFunctionEncoder(input_size=(0,), output_size=(2,), n_basis=n_basis, method=train_method).to(device)
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
    xs = None
    grid = torch.arange(-1, 1, 0.02, device=device)
    ys = torch.stack(torch.meshgrid(grid, grid), dim=-1).reshape(-1, 2).expand(10, -1, -1)
logits = model.predict_from_examples(example_xs, example_ys, xs, ys, method="grad_mle") # args.train_method)

with torch.no_grad():
    e_logits = torch.exp(logits)
    sums = torch.mean(e_logits, dim=1, keepdim=True) * dataset.volume
    pdf = e_logits / sums
    grid = grid.to("cpu").numpy()
    pdf = pdf.to("cpu").numpy()
    pdf = pdf.reshape(10, len(grid), len(grid))



    radii = info["radii"]
    for i in range(9):
        ax = axes[i ]
        ax.contourf(grid, grid, pdf[i], levels=100, cmap="Reds", )
        ax.scatter(example_ys[i, :, 0].cpu(), example_ys[i, :, 1].cpu(), color="black", s=1, alpha=0.5)
        circle = plt.Circle((0, 0), radii[i], color='b', fill=False)
        ax.add_artist(circle)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    # color bar
    cax = fig.add_subplot(gs[:, -1])
    cbar = plt.colorbar(ax.collections[0], cax=cax, orientation="vertical", fraction=0.1)

    plt.tight_layout()
    plt.savefig(f"{logdir}/donuts.png")
