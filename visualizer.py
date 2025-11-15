"""
Visualization module for classification and training convergence plots.

This module creates publication-quality visualizations of:
- Classification results with decision boundary
- Training convergence metrics over iterations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from model_trainer import sigmoid


def plot_classification(
    predictions_df: pd.DataFrame,
    coefficients: np.ndarray,
    output_file: str = 'classification_plot.png'
) -> None:
    """Create classification visualization plot.

    Plots ground truth as circles and predictions as X markers,
    with decision boundary shown as a contour line at sigmoid = 0.5.

    Args:
        predictions_df: DataFrame with predictions and ground truth
        coefficients: Model coefficients [B0, B1, B2]
        output_file: Output filename (default: 'classification_plot.png')
    """
    fig, ax = plt.subplots(figsize=(12.5, 8.5), dpi=100)

    # Separate by ground truth
    class_0 = predictions_df[predictions_df['R'] == 0]
    class_1 = predictions_df[predictions_df['R'] == 1]

    # Plot ground truth (circles)
    ax.scatter(class_0['x1'], class_0['x2'], c='blue', marker='o',
               s=100, alpha=0.6, label='Ground Truth: Class 0', edgecolors='black')
    ax.scatter(class_1['x1'], class_1['x2'], c='red', marker='o',
               s=100, alpha=0.6, label='Ground Truth: Class 1', edgecolors='black')

    # Plot predictions (X markers)
    pred_0 = predictions_df[predictions_df['sigma'] == 0]
    pred_1 = predictions_df[predictions_df['sigma'] == 1]

    ax.scatter(pred_0['x1'], pred_0['x2'], c='cyan', marker='x',
               s=80, linewidths=2, label='Predicted: Class 0')
    ax.scatter(pred_1['x1'], pred_1['x2'], c='orange', marker='x',
               s=80, linewidths=2, label='Predicted: Class 1')

    # Plot decision boundary
    x1_range = np.linspace(0, 1, 300)
    x2_range = np.linspace(0, 1, 300)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

    # Calculate sigmoid for all grid points
    Z = sigmoid(coefficients[0] + coefficients[1] * X1_grid + coefficients[2] * X2_grid)

    # Plot contour at 0.5 (decision boundary)
    contour = ax.contour(X1_grid, X2_grid, Z, levels=[0.5], colors='green', linewidths=2)
    ax.clabel(contour, inline=True, fontsize=10, fmt='Decision Boundary')

    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title('Binary Classification: Ground Truth vs Predictions', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()


def plot_training_convergence(
    training_history: List[Dict],
    output_file: str = 'training_convergence.png'
) -> None:
    """Create training convergence visualization.

    Plots both log-likelihood and MSE over iterations on dual Y-axes
    to show convergence progression during model training.

    Args:
        training_history: List of dicts with iteration, log_likelihood, mse
        output_file: Output filename (default: 'training_convergence.png')
    """
    iterations = [h['iteration'] for h in training_history]
    log_likelihoods = [h['log_likelihood'] for h in training_history]
    mses = [h['mse'] for h in training_history]

    fig, ax1 = plt.subplots(figsize=(12.5, 8.5), dpi=100)

    # Plot log-likelihood on left axis
    color = 'tab:blue'
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Log-Likelihood', color=color, fontsize=12)
    ax1.plot(iterations, log_likelihoods, color=color, linewidth=2, label='Log-Likelihood')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for MSE
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mean Squared Error', color=color, fontsize=12)
    ax2.plot(iterations, mses, color=color, linewidth=2, linestyle='--', label='MSE')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add title
    plt.title('Training Convergence: Log-Likelihood and MSE over Iterations', fontsize=14)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
