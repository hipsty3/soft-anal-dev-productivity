import os
from matplotlib.ticker import FuncFormatter
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

def percent_axis_x(ax, decimals=0):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{decimals}%}"))

def percent_axis_y(ax, decimals=0):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.{decimals}%}"))

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
    
def paired_violinplot(data, x, y1, y2, title1, title2, xlabel, ylabel1, ylabel2, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.violinplot(data=data, x=x, y=y1, ax=axes[0])
    axes[0].set_title(title1)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel1)

    sns.violinplot(data=data, x=x, y=y2, ax=axes[1])
    axes[1].set_title(title2)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel2)
    
    save_show(fig, save_path)