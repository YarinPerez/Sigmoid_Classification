"""
Logistic regression model training using custom gradient descent.

This module implements logistic regression with sigmoid activation,
tracking training metrics at each iteration for convergence visualization.
"""

import numpy as np
from typing import Tuple, List, Dict


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function with numerical stability.

    Maps real numbers to (0, 1) using the sigmoid function:
    σ(z) = 1 / (1 + e^(-z))

    Clips extreme values to prevent overflow/underflow.

    Args:
        z: Input array of any shape

    Returns:
        Sigmoid output in range (0, 1), same shape as input
    """
    # Clip for numerical stability
    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))


def compute_log_likelihood(
    y_true: np.ndarray,
    y_pred_continuous: np.ndarray
) -> float:
    """Compute log-likelihood for binary classification.

    Formula: Σ[R*log(σ) + (1-R)*log(1-σ)]

    Args:
        y_true: Ground truth labels (0 or 1), shape (n_samples,)
        y_pred_continuous: Continuous predictions from sigmoid, shape (n_samples,)

    Returns:
        Log-likelihood value (typically negative)
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred_safe = np.clip(y_pred_continuous, epsilon, 1 - epsilon)

    log_likelihood = np.sum(
        y_true * np.log(y_pred_safe) +
        (1 - y_true) * np.log(1 - y_pred_safe)
    )
    return float(log_likelihood)


def compute_mse(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray
) -> float:
    """Compute mean squared error for binary predictions.

    Formula: mean((σ - R)²)

    Args:
        y_true: Ground truth labels (0 or 1), shape (n_samples,)
        y_pred_binary: Binary predictions (0 or 1), shape (n_samples,)

    Returns:
        Mean squared error value in [0, 1]
    """
    mse = np.mean((y_pred_binary - y_true) ** 2)
    return float(mse)


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, List[Dict]]:
    """Train logistic regression using batch gradient descent.

    Implements logistic regression with batch gradient descent optimization.
    Tracks log-likelihood and MSE at each iteration for convergence analysis.

    Args:
        X: Feature matrix of shape (n_samples, 2) with values in [0, 1]
        y: Labels of shape (n_samples,) with values 0 or 1
        learning_rate: Learning rate for gradient descent (default: 0.1)
        max_iterations: Maximum training iterations (default: 1000)
        tolerance: Convergence tolerance (default: 1e-6)

    Returns:
        Tuple of:
        - coefficients: Array [B0, B1, B2] representing the learned parameters
        - training_history: List of dicts with keys: iteration, log_likelihood, mse

    Raises:
        ValueError: If inputs have invalid shape or values
    """
    # Validate inputs
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if X.shape[1] != 2:
        raise ValueError("X must have 2 features")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only 0 or 1")

    n_samples, n_features = X.shape

    # Add bias term (column of ones)
    X_with_bias = np.column_stack([np.ones(n_samples), X])

    # Initialize coefficients [B0, B1, B2]
    coefficients = np.zeros(n_features + 1)

    training_history = []

    for iteration in range(max_iterations):
        # Compute predictions
        z = X_with_bias @ coefficients
        y_pred_continuous = sigmoid(z)

        # Convert to binary for MSE calculation
        y_pred_binary = (y_pred_continuous >= 0.5).astype(int)

        # Compute metrics
        log_likelihood = compute_log_likelihood(y, y_pred_continuous)
        mse = compute_mse(y, y_pred_binary)

        # Store metrics
        training_history.append({
            'iteration': iteration,
            'log_likelihood': log_likelihood,
            'mse': mse
        })

        # Compute gradient: X^T @ (y_pred - y) / n_samples
        gradient = X_with_bias.T @ (y_pred_continuous - y) / n_samples

        # Update coefficients
        new_coefficients = coefficients - learning_rate * gradient

        # Check convergence: if change is smaller than tolerance, stop
        coefficient_change = np.linalg.norm(new_coefficients - coefficients)
        if coefficient_change < tolerance:
            coefficients = new_coefficients
            break

        coefficients = new_coefficients

    # Log convergence status
    if iteration == max_iterations - 1:
        print(f"Warning: Did not converge within {max_iterations} iterations")
    else:
        print(f"Converged at iteration {iteration}")

    return coefficients, training_history
