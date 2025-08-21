from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import ConnectionPatch

from FunctionEncoder import CIFARDataset, FunctionEncoder, ListCallback, TensorboardCallback, DistanceCallback

import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=100)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--grad_steps", type=int, default=1_000)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true")
args = parser.parse_args()


# hyper params
device = "cuda" if torch.cuda.is_available() else "cpu"
if args.load_path is None:
    logdir = f"logs/cifar_example/{args.train_method}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = args.load_path

# seed torch
torch.manual_seed(args.seed)

# create a dataset
train_dataset = CIFARDataset()
test_dataset = CIFARDataset(heldout_classes_only=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

# create the model
model = FunctionEncoder(input_size=train_dataset.input_size,
                        output_size=train_dataset.output_size,
                        data_type=train_dataset.data_type,
                        n_basis=args.n_basis,
                        method=args.train_method,
                        model_type="CNN",
                        use_residuals_method=args.residuals).to(device)

if args.load_path is None:
    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = DistanceCallback(test_dataloader, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(train_dataloader, grad_steps=args.grad_steps, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))











# plot

# create a plot with 4 rows, 9 columns
# each row is a different class
# the first 4 columns are positive examples.
# the second 4 columns are negative examples.
# the last column is a new image from the class and its prediction
def plot(example_xs, y_hats, info, dataset, logdir, filename,):
    fig, ax = plt.subplots(4, 12, figsize=(18, 8), gridspec_kw={'width_ratios': [1,1,1,1,0.2,1,1,1,1,0.2,1, 1]})
    for row in range(4):
        # positive examples
        for col in range(4):
            ax[row, col].axis("off")
            img = example_xs[row, col].permute(2,1,0).cpu().numpy()
            ax[row, col].imshow(img)
            class_idx = info["positive_example_class_indicies"][row]
            class_name = dataset.classes[class_idx]
            ax[row, col].set_title(class_name)

        # negative examples
        for col in range(5, 9):
            ax[row, col].axis("off")
            img = example_xs[row, -col + 4].permute(2,1,0).cpu().numpy()
            ax[row, col].imshow(img)
            class_idx = info["negative_example_class_indicies"][row, -col+4]
            class_name = dataset.classes[class_idx]
            ax[row, col].set_title(class_name)

        # disable axis for the two unfilled plots
        ax[row, 4].axis("off")
        ax[row, 9].axis("off")

        # new image and prediction
        ax[row, 10].axis("off")
        img = xs[row, 0].permute(2,1,0).cpu().numpy()
        ax[row, 10].imshow(img)

        logits = y_hats[row, 0]
        probs = torch.softmax(logits, dim=-1)
        ax[row, 10].set_title(f"$P(x \in C) = {probs[0].item()*100:.0f}\%$")

        # add new negative image and prediction
        ax[row, 11].axis("off")
        img = xs[row, -1].permute(2,1,0).cpu().numpy()
        ax[row, 11].imshow(img)

        logits = y_hats[row, -1]
        probs = torch.softmax(logits, dim=-1)
        ax[row, 11].set_title(f"$P(x \in C) = {probs[0].item()*100:.0f}\%$")

    # add dashed lines between positive and negative examples
    left = ax[0, 3].get_position().xmax
    right = ax[0, 5].get_position().xmin
    xpos = (left+right) / 2
    top = ax[0, 3].get_position().ymax + 0.05
    bottom = ax[3, 3].get_position().ymin
    line1 = matplotlib.lines.Line2D((xpos, xpos), (bottom, top),transform=fig.transFigure, color="black", linestyle="--")

    # add dashed lines between negative examples and new image
    left = ax[0, 8].get_position().xmax
    right = ax[0, 10].get_position().xmin
    xpos = (left+right) / 2
    line2 = matplotlib.lines.Line2D((xpos, xpos), (bottom, top),transform=fig.transFigure, color="black", linestyle="--")

    fig.lines = line1, line2,

    # add one text above positive samples
    left = ax[0, 0].get_position().xmin
    right = ax[0, 3].get_position().xmax
    xpos = (left+right) / 2
    ypos = ax[0, 0].get_position().ymax + 0.08
    fig.text(xpos, ypos, "Positive Examples", ha="center", va="center", fontsize=16, weight="bold")

    # add one text above negative samples
    left = ax[0, 5].get_position().xmin
    right = ax[0, 8].get_position().xmax
    xpos = (left+right) / 2
    fig.text(xpos, ypos, "Negative Examples", ha="center", va="center", fontsize=16, weight="bold")

    # add one text above new image
    left = ax[0, 10].get_position().xmin
    right = ax[0, 11].get_position().xmax
    xpos = (left+right) / 2
    fig.text(xpos, ypos, "New Image", ha="center", va="center", fontsize=16, weight="bold")


    plt.savefig(f"{logdir}/{filename}.png")
    plt.clf()

# get a new dataset for testing
dataset = CIFARDataset(split="test")

# ID test
example_xs, example_ys, xs, ys, info = next(iter(train_dataloader))
y_hats = model.predict_from_examples(example_xs, example_ys, xs)
plot(example_xs, y_hats, info, dataset, logdir, "in_distribution")

# OOD Test
example_xs, example_ys, xs, ys, info = next(iter(test_dataloader)) # heldout classes
y_hats = model.predict_from_examples(example_xs, example_ys, xs)
plot(example_xs, y_hats, info, dataset, logdir, "out_of_distribution")