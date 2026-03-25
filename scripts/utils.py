import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt


# -------------------------------
# Helper function
# -------------------------------
def interpret_coef(coef):
    return np.exp(coef) - 1


def save_show(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def paired_boxplot(data, x, y1, y2, title1, title2, xlabel, ylabel1, ylabel2, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.boxplot(data=data, x=x, y=y1, ax=axes[0])
    axes[0].set_title(title1)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel1)

    sns.boxplot(data=data, x=x, y=y2, ax=axes[1])
    axes[1].set_title(title2)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel2)

    save_show(fig, save_path)


def paired_boxplot_with_means(data, x, y1, y2, title1, title2, xlabel, ylabel1, ylabel2, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.boxplot(data=data, x=x, y=y1, ax=axes[0])
    sns.pointplot(data=data, x=x, y=y1, color="red", errorbar=None, ax=axes[0])
    axes[0].set_title(title1)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel1)

    sns.boxplot(data=data, x=x, y=y2, ax=axes[1])
    sns.pointplot(data=data, x=x, y=y2, color="red", errorbar=None, ax=axes[1])
    axes[1].set_title(title2)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel2)

    save_show(fig, save_path)


def paired_regplot(data, x, y1, y2, title1, title2, xlabel, ylabel1, ylabel2, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.regplot(
        data=data, x=x, y=y1,
        scatter_kws={"alpha": 0.3},
        line_kws={"color": "red"},
        ax=axes[0]
    )
    axes[0].set_title(title1)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel1)

    sns.regplot(
        data=data, x=x, y=y2,
        scatter_kws={"alpha": 0.3},
        line_kws={"color": "red"},
        ax=axes[1]
    )
    axes[1].set_title(title2)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel2)

    save_show(fig, save_path)


def paired_lineplot_from_summary(x_vals, y1_vals, y2_vals, title1, title2, xlabel, ylabel1, ylabel2, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x_vals, y1_vals, marker="o")
    axes[0].set_title(title1)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel1)

    axes[1].plot(x_vals, y2_vals, marker="o")
    axes[1].set_title(title2)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel2)

    save_show(fig, save_path)