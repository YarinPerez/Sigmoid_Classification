"""
Dataset generation for binary classification.

This module generates synthetic 2D clustered data for binary classification.
Features are normalized to [0, 1] range.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_dataset(
    n_samples_per_cluster: int = 100,
    cluster_1_center: Tuple[float, float] = (0.3, 0.3),
    cluster_2_center: Tuple[float, float] = (0.7, 0.7),
    cluster_std: float = 0.1,
    random_seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic 2D clustered dataset for binary classification.

    Creates two Gaussian clusters with configurable centers and spread.
    All values are clipped to [0, 1] range.

    Args:
        n_samples_per_cluster: Number of points per cluster (default: 100)
        cluster_1_center: (x1, x2) center for cluster 0 (default: (0.3, 0.3))
        cluster_2_center: (x1, x2) center for cluster 1 (default: (0.7, 0.7))
        cluster_std: Standard deviation for both clusters (default: 0.1)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        DataFrame with columns: x1, x2, label

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate inputs
    if n_samples_per_cluster <= 0:
        raise ValueError("n_samples_per_cluster must be positive")
    if cluster_std <= 0:
        raise ValueError("cluster_std must be positive")

    np.random.seed(random_seed)

    # Generate cluster 1 (label = 0)
    cluster_1 = np.random.normal(
        loc=cluster_1_center,
        scale=cluster_std,
        size=(n_samples_per_cluster, 2)
    )

    # Generate cluster 2 (label = 1)
    cluster_2 = np.random.normal(
        loc=cluster_2_center,
        scale=cluster_std,
        size=(n_samples_per_cluster, 2)
    )

    # Combine clusters
    data = np.vstack([cluster_1, cluster_2])

    # Track clipping
    original_data = data.copy()
    data = np.clip(data, 0, 1)

    # Warn if significant clipping
    clipped_count = np.sum(data != original_data)
    total_values = data.size
    clipped_percentage = (clipped_count / total_values) * 100

    if clipped_percentage > 5:
        print(f"Warning: {clipped_percentage:.2f}% of points were clipped to [0, 1] range")

    # Create labels: 0 for cluster 1, 1 for cluster 2
    labels = np.array([0] * n_samples_per_cluster + [1] * n_samples_per_cluster)

    # Create DataFrame
    df = pd.DataFrame({
        'x1': data[:, 0],
        'x2': data[:, 1],
        'label': labels
    })

    return df
