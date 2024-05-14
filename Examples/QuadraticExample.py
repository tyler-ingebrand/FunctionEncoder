from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import FunctionEncoder, QuadraticDataset
from FunctionEncoder.Callbacks.TestPerformanceCallback import TestPerformanceCallback

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="inner_product")
parser.add_argument("--epochs", type=int, default=1_000)
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
    logdir = f"logs/quadratic_example/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

# create a dataset
a_range = (-3, 3)
b_range = (-3, 3)
c_range = (-3, 3)
input_range = (-10, 10)
dataset = QuadraticDataset(a_range=a_range, b_range=b_range, c_range=c_range, input_range=input_range)

if load_path is None:
    # create the model
    model = FunctionEncoder(input_size=(1,), output_size=(1,), n_basis=n_basis, method=train_method).to(device)

    # create a testing callback
    callback = TestPerformanceCallback(dataset, device=device)

    # train the model
    model.train_model(dataset, epochs=epochs, logdir=logdir, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model = FunctionEncoder(input_size=(1,), output_size=(1,), n_basis=n_basis, method=train_method).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# plot
with torch.no_grad():
    n_plots = 9
    n_examples = 100
    example_xs, example_ys, xs, ys, info = dataset.sample(device)
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]
    y_hats_ls = model.predict_from_examples(example_xs, example_ys, xs, method="least_squares")
    y_hats_ip = model.predict_from_examples(example_xs, example_ys, xs, method="inner_product")
    xs, indicies = torch.sort(xs, dim=-2)
    ys = ys.gather(dim=-2, index=indicies)
    y_hats_ls = y_hats_ls.gather(dim=-2, index=indicies)
    y_hats_ip = y_hats_ip.gather(dim=-2, index=indicies)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        ax.plot(xs[i].cpu(), ys[i].cpu(), label="True")
        ax.plot(xs[i].cpu(), y_hats_ls[i].cpu(), label="LS")
        ax.plot(xs[i].cpu(), y_hats_ip[i].cpu(), label="IP")
        if i == n_plots - 1:
            ax.legend()
        title = f"${info['As'][i].item():.2f}x^2 + {info['Bs'][i].item():.2f}x + {info['Cs'][i].item():.2f}$"
        ax.set_title(title)
        y_min, y_max = ys[i].min().item(), ys[i].max().item()
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.clf()
    # print corresponding MSEs
    mse_ls = ((ys - y_hats_ls) ** 2).mean()
    mse_ip = ((ys - y_hats_ip) ** 2).mean()
    print(f"MSE LS: {mse_ls.item()}")
    print(f"MSE IP: {mse_ip.item()}")


    # plot the basis functions
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    xs = torch.linspace(input_range[0], input_range[1], 1_000).unsqueeze(-1).to(device)
    basis = model.forward(xs)
    for i in range(n_basis):
        ax.plot(xs.cpu(), basis[:, 0, i].cpu())
    plt.tight_layout()
    plt.savefig(f"{logdir}/basis.png")


