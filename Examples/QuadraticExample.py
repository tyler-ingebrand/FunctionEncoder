from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import QuadraticDataset, FunctionEncoder, MSECallback, ListCallback, TensorboardCallback, \
    DistanceCallback

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--grad_steps", type=int, default=1_000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
parser.add_argument("--parallel", action="store_true")
args = parser.parse_args()


# hyper params
device = "cuda" if torch.cuda.is_available() else "cpu"
if args.load_path is None:
    logdir = f"logs/quadratic_example/{args.train_method}/{'shared_model' if not args.parallel else 'parallel_models'}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = args.load_path
arch = "MLP" if not args.parallel else "ParallelMLP"

# seed torch
torch.manual_seed(args.seed)

# create a dataset
if args.residuals:
    a_range = (0, 3/50) # this makes the true average function non-zero
else:
    a_range = (-3/50, 3/50)
b_range = (-3/50, 3/50)
c_range = (-3/50, 3/50)
input_range = (-10, 10)
dataset = QuadraticDataset(a_range=a_range, b_range=b_range, c_range=c_range, input_range=input_range)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# create the function encoder
model = FunctionEncoder(input_size=dataset.input_size,
                        output_size=dataset.output_size,
                        data_type=dataset.data_type,
                        n_basis=args.n_basis,
                        model_type=arch,
                        method=args.train_method,
                        use_residuals_method=args.residuals).to(device)
print('Number of parameters:', sum(p.numel() for p in model.parameters()))


if args.load_path is None:
    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = DistanceCallback(dataloader, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(dataloader, grad_steps=args.grad_steps, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))








# plot.
with torch.no_grad():
    n_plots = 9
    n_examples = 100
    example_xs, example_ys, query_xs, query_ys, info = next(iter(dataloader))
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]
    if args.train_method == "inner_product":
        y_hats_ip = model.predict_from_examples(example_xs, example_ys, query_xs, method="inner_product")
    y_hats_ls = model.predict_from_examples(example_xs, example_ys, query_xs, method="least_squares")
    query_xs, indicies = torch.sort(query_xs, dim=-2)
    query_ys = query_ys.gather(dim=-2, index=indicies)
    y_hats_ls = y_hats_ls.gather(dim=-2, index=indicies)
    if args.train_method == "inner_product":
        y_hats_ip = y_hats_ip.gather(dim=-2, index=indicies)

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        ax.plot(query_xs[i].cpu(), query_ys[i].cpu(), label="True")
        ax.plot(query_xs[i].cpu(), y_hats_ls[i].cpu(), label="LS")
        if args.train_method == "inner_product":
            ax.plot(query_xs[i].cpu(), y_hats_ip[i].cpu(), label="IP")
        if i == n_plots - 1:
            ax.legend()
        title = f"${info['As'][i].item():.2f}x^2 + {info['Bs'][i].item():.2f}x + {info['Cs'][i].item():.2f}$"
        ax.set_title(title)
        y_min, y_max = query_ys[i].min().item(), query_ys[i].max().item()
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.clf()

    # plot the basis functions
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    query_xs = torch.linspace(input_range[0], input_range[1], 1_000).reshape(1000, 1).to(device)
    basis = model.forward_basis_functions(query_xs)
    for i in range(args.n_basis):
        ax.plot(query_xs.flatten().cpu(), basis[:, 0, i].cpu(), color="black")
    if args.residuals:
        avg_function = model.average_function.forward(query_xs)
        ax.plot(query_xs.flatten().cpu(), avg_function.flatten().cpu(), color="blue")

    plt.tight_layout()
    plt.savefig(f"{logdir}/basis.png")


