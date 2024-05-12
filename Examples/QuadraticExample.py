from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import FunctionEncoder, QuadraticDataset

# hyper params
epochs = 1000
n_basis = 11
device = "cuda"

# create a dataset
a_range = (-3, 3)
b_range = (-3, 3)
c_range = (-3, 3)
input_range = (-10, 10)
dataset = QuadraticDataset(a_range=a_range, b_range=b_range, c_range=c_range, input_range=input_range)

# create the model
model = FunctionEncoder(input_size=(1,), output_size=(1,), n_basis=n_basis).to(device)

# train the model
date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = f"logs/quadratic_example/{date_time_str}"
model.train_model(dataset, epochs=epochs, logdir=logdir)

# save the model
torch.save(model.state_dict(), f"{logdir}/model.pth")

# plot
with torch.no_grad():
    n_plots = 9
    n_examples = 10
    example_xs, example_ys, xs, ys, info = dataset.sample(device)
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]
    y_hats_ls = model.predict_from_examples(example_xs, example_ys, xs, method="least_squares")
    y_hats_ip = model.predict_from_examples(example_xs, example_ys, xs, method="inner_product")
    xs, indicies = torch.sort(xs, dim=-2)
    ys = ys.gather(dim=-2, index=indicies)
    y_hats_ls = y_hats_ls.gather(dim=-2, index=indicies)
    y_hats_ip = y_hats_ip.gather(dim=-2, index=indicies)

    # x_min, x_max = input_range
    # y_min, y_max = a_range[0] * x_min ** 2 + b_range[0] * x_min + c_range[0], a_range[1] * x_max ** 2 + b_range[1] * x_max + c_range[1]

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
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")

    # print corresponding MSEs
    mse_ls = ((ys - y_hats_ls) ** 2).mean()
    mse_ip = ((ys - y_hats_ip) ** 2).mean()
    print(f"MSE LS: {mse_ls.item()}")
    print(f"MSE IP: {mse_ip.item()}")

