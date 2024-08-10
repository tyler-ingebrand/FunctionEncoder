from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import CategoricalDataset, FunctionEncoder, ListCallback, TensorboardCallback, DistanceCallback

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
args = parser.parse_args()


# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
train_method = args.train_method
seed = args.seed
load_path = args.load_path
residuals = args.residuals
if load_path is None:
    logdir = f"logs/categorical_example/{train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path

# seed torch
torch.manual_seed(seed)

# create a dataset
dataset = CategoricalDataset()

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
    cb2 = DistanceCallback(dataset, device=device, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# plot
with torch.no_grad():
    n_plots = 9
    n_examples = 100

    # SAMPLE DATA
    example_xs, example_ys, xs, ys, info = dataset.sample()
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]

    # get predictions
    xs, indicies = torch.sort(xs, dim=-2)
    y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=args.train_method)

    # get ground truth
    most_likely_category = y_hats.argmax(dim=2, keepdim=True)
    boundaries = info["boundaries"]
    ground_truth_categories = info["categories"]

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]

        # plot dashed lines at the boundaries
        label = "Groundtruth Boundary"
        for b in boundaries[i]:
            ax.axvline(b.item(), color="black", linestyle="--", label=label)
            label=None

        # add text labeling the sections with A, B, C, etc
        boundaries_i = boundaries[i]
        boundaries_i = torch.cat([torch.tensor([0]), boundaries_i.to("cpu"), torch.tensor([1])])
        for j in range(len(boundaries_i) - 1):
            a, b = boundaries_i[j], boundaries_i[j + 1]
            if torch.abs(a-b).item() > 0.05:
                ax.text((a.item() + b.item()) / 2, 1.5, chr(65 + ground_truth_categories[i][j]), fontsize=30, verticalalignment="center", horizontalalignment="center")

        # plot predictions
        ax.plot(xs[i].cpu(), most_likely_category[i].cpu(), label="Predicted Category")

        # legend
        if i == n_plots - 1:
            ax.legend()

        # change yaxis labels to A, B, C, etc
        ax.set_yticks(range(dataset.n_categories))
        ax.set_yticklabels([chr(65 + j) for j in range(dataset.n_categories)])

        # y_min, y_max = ys[i].min().item(), ys[i].max().item()
        # ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.clf()



