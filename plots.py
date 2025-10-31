import matplotlib.pyplot as plt
import numpy as np


def heatmap_lambda_gamma(cm, path, class_names=("0", "1")):
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_curve(x, y, path, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
