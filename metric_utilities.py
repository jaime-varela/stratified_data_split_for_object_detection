from collections import Counter
from scipy.stats import entropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def compute_class_distribution(coco_obj, class_names):
    """Returns a normalized class distribution: class_name -> frequency"""
    
    # Build mapping from category_id → class name using the COCO categories list
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_obj["categories"] if cat["name"] in class_names}
    
    class_counts = Counter()
    for ann in coco_obj["annotations"]:
        category_id = ann["category_id"]
        class_name = cat_id_to_name.get(category_id)
        if class_name is not None:
            class_counts[class_name] += 1

    total = sum(class_counts.values())
    return {cls: count / total for cls, count in class_counts.items()}


def compute_deviation_metrics(global_dist, split_dist,
                              entropy_normalization = 1e-8):
    """
    Computes deviation metrics between two class distributions.
    Returns: dict with keys 'l1', 'kl', 'max_abs'
    """
    classes = sorted(global_dist.keys())
    g = np.array([global_dist[c] for c in classes])
    s = np.array([split_dist.get(c, 0.0) for c in classes])  # Fill in missing with 0

    metrics = {
        "l1": np.sum(np.abs(g - s)),
        "kl": entropy(s + entropy_normalization,g + entropy_normalization), # g is in the p log(p/q) term because g is guaranteed to have nonzero counts
        "max_abs": np.max(np.abs(g - s)),
    }
    return metrics


def compute_split_ratio_deviation(coco_per_assignment, split_ratios, assignment_index_to_name, total_images):
    """
    Computes L1 deviation between actual and expected split ratios.
    Returns a dict: {split_name: deviation_from_expected}
    """
    expected = {assignment_index_to_name[i]: r for i, r in enumerate(split_ratios)}
    actual_counts = {split_name: len(split["images"]) for split_name, split in coco_per_assignment.items()}
    actual_ratios = {k: v / total_images for k, v in actual_counts.items()}

    deviation = {k: abs(actual_ratios.get(k, 0.0) - expected[k]) for k in expected}
    return deviation


def compute_distribution_from_df(df: pd.DataFrame) -> dict[str, float]:
    """
    Computes normalized class distribution from a DataFrame.
    Assumes first column is image ID, remaining columns are class counts.
    """
    class_columns = df.columns[1:]
    class_totals = df[class_columns].sum()
    total = class_totals.sum()
    return {cls: count / total for cls, count in class_totals.items()}


def compute_ratio_deviation(n_total, n_split, split_ratio):
    """Computes the absolute deviation between actual and expected split ratio."""
    actual = n_split / n_total
    return abs(actual - split_ratio)




# plotting function



def plot_metric_vs_image_size(
    all_metrics,
    metric_name,
    split_name,
    central_tendency='mean',
    spread='std',
    label=None,
    ax=None,
    show=True
):
    """
    Plots a dotted line for mean/median of a given metric with asymmetric vertical bars.

    Parameters:
        all_metrics: dict
            {(image_size, split_name): {metric_name: [values]}}
        metric_name: str
        central_tendency: 'mean' or 'median'
        spread: 'std' or 'max'
        split_name: str or None
            If provided, restrict to this split.
        label: str
            Label for the plot (used when overlaying)
        ax: matplotlib Axes object to draw on (for overlay)
        show: bool
            Whether to call plt.show() (default True)
    """
    assert central_tendency in {'mean', 'median'}, "central_tendency must be 'mean' or 'median'"
    assert spread in {'std', 'max'}, "spread must be 'std' or 'max'"

    # Aggregate metrics per image size and split
    data = []
    for (img_size, split), metrics in all_metrics.items():
        if metric_name not in metrics:
            continue
        if split_name and split != split_name:
            continue

        values = np.array(metrics[metric_name])
        center = values.mean() if central_tendency == 'mean' else np.median(values)
        if spread == 'std':
            spread_val = values.std()
        else:
            spread_val = values.max() - values.min()

        lower = min(center, spread_val)  # clamp to avoid going below zero
        data.append((img_size, center, lower, spread_val))

    # Sort and unpack
    data.sort(key=lambda x: x[0])
    sizes, centers, lowers, uppers = zip(*data) if data else ([], [], [], [])

    # Compute asymmetric error bars
    yerr = np.array([lowers, uppers])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(sizes, centers, yerr=yerr, fmt='o--', capsize=5, label=label)

    if show:
        ax.set_xlabel("Image Size (Number of Images)")
        ax.set_ylabel(f"{metric_name.upper()} ({central_tendency} ± {spread})")
        ax.set_title(f"{metric_name.upper()} vs Image Size" + (f" (Split: {split_name})" if split_name else ""))
        ax.grid(True, linestyle='--', alpha=0.6)
        if label:
            ax.legend()
        plt.tight_layout()
        plt.show()

    return ax



def compare_metric_overlay(all_metrics_1, all_metrics_2,
                           label_1: str, label_2: str,
                           split_name: str,
                           metric_name: str,
                           central_tendency: str = 'mean',
                           spread: str = 'std'):
    """
    Overlays metric-vs-image-size plots from two metric sources for a single split.

    Uses `plot_metric_vs_image_size` internally.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_metric_vs_image_size(
        all_metrics_1,
        metric_name=metric_name,
        central_tendency=central_tendency,
        spread=spread,
        split_name=split_name,
        label=label_1,
        ax=ax,
        show=False
    )

    plot_metric_vs_image_size(
        all_metrics_2,
        metric_name=metric_name,
        central_tendency=central_tendency,
        spread=spread,
        split_name=split_name,
        label=label_2,
        ax=ax,
        show=False
    )

    ax.set_xlabel("Image Size (Number of Images)")
    ax.set_ylabel(f"{metric_name.upper()} ({central_tendency} ± {spread})")
    ax.set_title(f"{metric_name.upper()} vs Image Size (Split: {split_name})")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()
