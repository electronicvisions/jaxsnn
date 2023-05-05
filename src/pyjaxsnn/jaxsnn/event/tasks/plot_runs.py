import matplotlib.pyplot as plt
import jax.numpy as np

import json
from pathlib import Path


def load_runs(folder):
    runs = []
    pathlist = Path(folder).glob("**/*")
    for path in pathlist:
        path_in_str = str(path)
        if path_in_str.endswith(".json"):
            with open(path) as file:
                data = json.load(file)
            runs.append(data)
    print(f"Found {len(runs)} runs")
    return runs


def plt_runs(folder):
    runs = load_runs(folder)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    # ax1.grid()
    # ax2.grid()
    ax2.axhline(y=0.9850, c="black", ls="dashed", alpha=0.7, label="98.50 %")
    ax2.text(46, 0.985, "98.50 %")

    n_epochs = len(runs[0]["loss"])
    mean_val_loss = np.zeros(n_epochs)
    x = np.arange(n_epochs)
    mean_acc = np.zeros(n_epochs)

    for run in runs:
        mean_val_loss += np.array(run["loss"])
        mean_acc += np.array(run["accuracy"])

        ax1.plot(x, run["loss"], c="gray", alpha=0.2)
        ax2.plot(x, run["accuracy"], c="gray", alpha=0.2)
    mean_val_loss /= len(runs)
    ax1.plot(x, mean_val_loss, c="blue", alpha=0.7)
    ax1.set_ylabel("Test loss")
    ax1.spines.right.set_visible(False)
    ax1.spines.top.set_visible(False)

    mean_acc /= len(runs)
    ax2.plot(x, mean_acc, c="blue", alpha=0.7)
    print(np.max(mean_acc))
    ax2.set_ylabel("Test accuracy")
    ax2.set_ylim(0.55, 1.05)
    ax2.set_xlabel("epochs")
    ax2.set_xlim(-1.5, 300.5)
    ax2.spines.right.set_visible(False)
    ax2.spines.top.set_visible(False)

    fig.tight_layout()
    plt.savefig(f"{folder}/different_seeds.png", dpi=300, transparent=False)


if __name__ == "__main__.py":
    folder = "../plots/event/yinyang/2023-01-28 12:33:13"
    plt_runs(folder)
