"""
Plotting utilities for the IDCR framework.

Provides consistent styling and common plot types for experiment visualisation.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Any
from pathlib import Path


# ── Global style ─────────────────────────────────────────────────────────────

def set_style():
    """Set consistent plotting style."""
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
    })


# ── Plot types ───────────────────────────────────────────────────────────────

def plot_marginal_gains(
    gains: np.ndarray,
    std: np.ndarray | None = None,
    title: str = "Marginal Gains",
    save_path: str | None = None,
):
    """Plot marginal gain curve with optional error bands.

    Args:
        gains: (k,) mean marginal gains.
        std: (k,) standard deviation.
        title: plot title.
        save_path: if set, save figure to path.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(1, len(gains) + 1)

    if std is not None:
        ax.fill_between(steps, gains - std, gains + std,
                        alpha=0.25, color="steelblue")
    ax.plot(steps, gains, "o-", color="steelblue", markersize=4)
    ax.set_xlabel("Step $|S|$")
    ax.set_ylabel("Marginal Gain $\\Delta(d|S)$")
    ax.set_title(title)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_coverage_vs_alpha(
    alphas: list[float],
    coverages: dict[str, list[float]],
    title: str = "Coverage vs α",
    save_path: str | None = None,
):
    """Plot empirical coverage vs target coverage for multiple methods.

    Args:
        alphas: list of α values.
        coverages: dict method_name → list of coverage values.
        title: plot title.
        save_path: save path.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Ideal line
    targets = [1.0 - a for a in alphas]
    ax.plot(alphas, targets, "k--", linewidth=2, label="Target (1-α)")

    for method, covs in coverages.items():
        ax.plot(alphas, covs, "o-", markersize=6, label=method)

    ax.set_xlabel("Miscoverage level $\\alpha$")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title(title)
    ax.legend()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_pareto_frontier(
    coverages: np.ndarray,
    volumes: np.ndarray,
    frontier_coverages: np.ndarray,
    frontier_volumes: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Pareto Frontier",
    save_path: str | None = None,
):
    """Plot efficiency-coverage Pareto frontier.

    Args:
        coverages: (n,) all coverage values.
        volumes: (n,) all volume values.
        frontier_coverages: Pareto-optimal coverages.
        frontier_volumes: Pareto-optimal volumes.
        labels: optional per-point labels.
        title: plot title.
        save_path: save path.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(coverages, volumes, alpha=0.4, color="gray", s=20, label="All configs")
    ax.plot(frontier_coverages, frontier_volumes, "ro-", markersize=6,
            linewidth=2, label="Pareto frontier")

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Set Volume")
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()  # Smaller volume is better

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_uncertainty_trajectory(
    log_dets: np.ndarray,
    per_dim_std: np.ndarray | None = None,
    asset_classes: list[str] | None = None,
    title: str = "Uncertainty Trajectory",
    save_path: str | None = None,
):
    """Plot how uncertainty evolves through retrieval.

    Args:
        log_dets: (k+1,) log-det at each step.
        per_dim_std: (k+1, d) per-dimension std devs.
        asset_classes: names for per-dimension plot.
        title: plot title.
        save_path: save path.
    """
    set_style()
    n_plots = 2 if per_dim_std is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Log-det trajectory
    ax = axes[0]
    steps = np.arange(len(log_dets))
    ax.plot(steps, log_dets, "o-", color="steelblue", markersize=5)
    ax.set_xlabel("Retrieval Step")
    ax.set_ylabel("$\\log \\det \\Sigma$")
    ax.set_title("Overall Uncertainty")

    # Per-dimension
    if per_dim_std is not None and len(axes) > 1:
        ax = axes[1]
        d = per_dim_std.shape[1]
        if asset_classes is None:
            asset_classes = [f"Dim {i}" for i in range(d)]
        for dim in range(d):
            ax.plot(steps, per_dim_std[:, dim], "-", alpha=0.7,
                    label=asset_classes[dim])
        ax.set_xlabel("Retrieval Step")
        ax.set_ylabel("Std Dev")
        ax.set_title("Per-Asset Uncertainty")
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_method_comparison(
    methods: list[str],
    metrics: dict[str, list[float]],
    title: str = "Method Comparison",
    save_path: str | None = None,
):
    """Bar chart comparing methods across metrics.

    Args:
        methods: list of method names.
        metrics: dict metric_name → list of values (one per method).
        title: plot title.
        save_path: save path.
    """
    set_style()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = sns.color_palette("viridis", len(methods))

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(methods, values, color=colors)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()
