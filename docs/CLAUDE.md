# Claude Implementation Guide: Sigmoid Binary Classification System

This document provides step-by-step instructions for an AI assistant (Claude or similar) to implement the Sigmoid Binary Classification System.

---

## Overview for AI Assistant

You are tasked with implementing a complete binary classification system using logistic regression with sigmoid activation. This guide provides:
1. Step-by-step implementation instructions
2. Code templates and examples
3. Common pitfalls to avoid
4. Testing checkpoints
5. Quality assurance guidelines

**Before starting**: Read PRD.md, PLANNING.md, and TODO.md to understand the full context.

---

## Implementation Workflow

### Pre-Implementation Checklist

Before writing any code:
- [ ] Read and understand PRD.md completely
- [ ] Review PLANNING.md for architecture decisions
- [ ] Understand all 4 functional requirements (FR-1.1 through FR-4.3)
- [ ] Note the 250-line limit per file
- [ ] Identify the 5 required output files
- [ ] Understand the training metrics tracking requirement

---

## Step-by-Step Implementation Guide

### STEP 0: Environment Setup

**Task**: Set up project structure and dependencies

**Instructions**:
1. Create `pyproject.toml` with dependencies:
   ```toml
   [project]
   name = "sigmoid-classifier"
   version = "1.0.0"
   requires-python = ">=3.8"
   dependencies = [
       "numpy>=1.20.0",
       "pandas>=1.3.0",
       "matplotlib>=3.4.0",
   ]
   ```

2. Create basic file structure:
   - main.py
   - data_generator.py
   - model_trainer.py
   - analyzer.py

3. In main.py, define configuration dictionary:
   ```python
   config = {
       'n_samples_per_cluster': 100,
       'cluster_1_center': (0.3, 0.3),
       'cluster_2_center': (0.7, 0.7),
       'cluster_std': 0.1,
       'random_seed': 42,
       'learning_rate': 0.1,
       'max_iterations': 1000,
       'convergence_tolerance': 1e-6
   }
   ```

**Verification**:
- Files created
- pyproject.toml is valid
- Configuration dictionary is complete

---

### STEP 1: Data Generation (FR-1.1)

**Task**: Implement dataset generation in `data_generator.py`

**Key Requirements**:
- Generate two clusters with normal distribution
- Clip values to [0, 1] range
- Warn if >5% of points are clipped
- Binary labels: 0 and 1
- Save to dataset.csv

**Implementation Template**:
```python
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

    Args:
        n_samples_per_cluster: Number of points per cluster
        cluster_1_center: (x1, x2) center for cluster 0
        cluster_2_center: (x1, x2) center for cluster 1
        cluster_std: Standard deviation for both clusters
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: x1, x2, label
    """
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
    clipped_percentage = (np.sum(data != original_data) / data.size) * 100
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
```

**Pitfalls to Avoid**:
- ❌ Don't forget to set random seed for reproducibility
- ❌ Don't use labels other than 0 and 1 (must be binary)
- ❌ Don't forget to check clipping percentage
- ❌ Don't forget to save to CSV with headers

**Verification**:
```python
# Test the function
df = generate_dataset()
assert df.shape == (200, 3), "Should have 200 rows, 3 columns"
assert set(df['label'].unique()) == {0, 1}, "Labels must be 0 and 1"
assert df['x1'].min() >= 0 and df['x1'].max() <= 1, "x1 must be in [0, 1]"
assert df['x2'].min() >= 0 and df['x2'].max() <= 1, "x2 must be in [0, 1]"
df.to_csv('dataset.csv', index=False)
print("✓ Step 1 complete: dataset.csv created")
```

---

### STEP 2: Model Training (FR-2.1)

**Task**: Implement custom logistic regression with iteration tracking in `model_trainer.py`

**Critical Requirements**:
- **MUST track metrics at each iteration** (log-likelihood and MSE)
- Use sigmoid function: σ(z) = 1 / (1 + e^(-z))
- Implement gradient descent
- Return training history as list of dictionaries

**Implementation Template**:
```python
import numpy as np
from typing import Tuple, List, Dict

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute sigmoid function with numerical stability.

    Args:
        z: Input array

    Returns:
        Sigmoid values in (0, 1)
    """
    # Clip for numerical stability
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_log_likelihood(y_true: np.ndarray, y_pred_continuous: np.ndarray) -> float:
    """Compute log-likelihood for binary classification.

    Formula: Σ[R*log(σ) + (1-R)*log(1-σ)]

    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred_continuous: Continuous predictions from sigmoid (0 to 1)

    Returns:
        Log-likelihood value
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred_continuous = np.clip(y_pred_continuous, epsilon, 1 - epsilon)

    log_likelihood = np.sum(
        y_true * np.log(y_pred_continuous) +
        (1 - y_true) * np.log(1 - y_pred_continuous)
    )
    return log_likelihood

def compute_mse(y_true: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """Compute mean squared error for binary predictions.

    Args:
        y_true: Ground truth (0 or 1)
        y_pred_binary: Binary predictions (0 or 1)

    Returns:
        Mean squared error
    """
    return np.mean((y_pred_binary - y_true) ** 2)

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    max_iterations: int = 1000,
    tolerance: float = 1e-6
) -> Tuple[np.ndarray, List[Dict]]:
    """Train logistic regression using gradient descent.

    Args:
        X: Feature matrix (n_samples, 2)
        y: Labels (n_samples,) with values 0 or 1
        learning_rate: Learning rate for gradient descent
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of:
        - coefficients: [B0, B1, B2]
        - training_history: List of {'iteration', 'log_likelihood', 'mse'} dicts
    """
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

        # Compute gradient
        gradient = X_with_bias.T @ (y_pred_continuous - y) / n_samples

        # Update coefficients
        new_coefficients = coefficients - learning_rate * gradient

        # Check convergence
        if np.linalg.norm(new_coefficients - coefficients) < tolerance:
            print(f"Converged at iteration {iteration}")
            coefficients = new_coefficients
            break

        coefficients = new_coefficients

    if iteration == max_iterations - 1:
        print(f"Warning: Did not converge within {max_iterations} iterations")

    return coefficients, training_history
```

**Pitfalls to Avoid**:
- ❌ Don't forget to add bias term (column of ones) to X
- ❌ Don't skip numerical stability (clip z values, add epsilon to log)
- ❌ Don't forget to store metrics at EVERY iteration
- ❌ Don't use MSE on continuous predictions (must be binary: 0 or 1)
- ❌ Don't forget to check convergence

**Verification**:
```python
# Load dataset
df = pd.read_csv('dataset.csv')
X = df[['x1', 'x2']].values
y = df['label'].values

# Train model
coefficients, training_history = train_logistic_regression(X, y)

# Verify
assert len(coefficients) == 3, "Should have 3 coefficients (B0, B1, B2)"
assert len(training_history) > 0, "Should have training history"
assert all('log_likelihood' in h for h in training_history), "Missing log_likelihood"
assert all('mse' in h for h in training_history), "Missing mse"
print(f"✓ Step 2 complete: Model trained with {len(training_history)} iterations")
print(f"  Final log-likelihood: {training_history[-1]['log_likelihood']:.2f}")
print(f"  Final MSE: {training_history[-1]['mse']:.4f}")
```

---

### STEP 3: Prediction and Error Calculation (FR-3.1)

**Task**: Generate predictions and calculate errors in `analyzer.py` (or separate module)

**Key Requirements**:
- Calculate continuous sigmoid output
- Convert to binary (threshold at 0.5)
- Calculate squared error: (σ - R)²
- Save to predictions.csv

**Implementation Template**:
```python
def predict_and_calculate_errors(
    X: np.ndarray,
    y_true: np.ndarray,
    coefficients: np.ndarray
) -> pd.DataFrame:
    """Generate predictions and calculate squared errors.

    Args:
        X: Feature matrix (n_samples, 2)
        y_true: Ground truth labels (0 or 1)
        coefficients: Model coefficients [B0, B1, B2]

    Returns:
        DataFrame with columns: x1, x2, R, sigma_continuous, sigma, error_squared
    """
    # Add bias term
    X_with_bias = np.column_stack([np.ones(len(X)), X])

    # Calculate continuous predictions
    z = X_with_bias @ coefficients
    sigma_continuous = sigmoid(z)

    # Convert to binary
    sigma = (sigma_continuous >= 0.5).astype(int)

    # Calculate squared error
    error_squared = (sigma - y_true) ** 2

    # Create DataFrame
    predictions_df = pd.DataFrame({
        'x1': X[:, 0],
        'x2': X[:, 1],
        'R': y_true,
        'sigma_continuous': sigma_continuous,
        'sigma': sigma,
        'error_squared': error_squared
    })

    return predictions_df
```

**Pitfalls to Avoid**:
- ❌ Don't forget bias term in prediction
- ❌ Don't use wrong threshold (must be 0.5)
- ❌ Don't calculate error on continuous values (must be binary)
- ❌ Error formula is (σ - R)², not |σ - R|

**Verification**:
```python
predictions_df = predict_and_calculate_errors(X, y, coefficients)
assert set(predictions_df['R'].unique()).issubset({0, 1}), "R must be 0 or 1"
assert set(predictions_df['sigma'].unique()).issubset({0, 1}), "sigma must be 0 or 1"
assert set(predictions_df['error_squared'].unique()).issubset({0, 1}), "error² must be 0 or 1"
predictions_df.to_csv('predictions.csv', index=False)
print("✓ Step 3 complete: predictions.csv created")
```

---

### STEP 4.1: Classification Visualization (FR-4.1)

**Task**: Create scatter plot showing ground truth, predictions, and decision boundary

**Key Requirements**:
- Ground truth: circles (different colors for 0 vs 1)
- Predictions: X markers (different colors for 0 vs 1)
- Decision boundary: contour line where σ_continuous = 0.5
- Resolution: 1200x800 minimum

**Implementation Template**:
```python
import matplotlib.pyplot as plt

def plot_classification(predictions_df: pd.DataFrame, coefficients: np.ndarray):
    """Create classification visualization plot.

    Args:
        predictions_df: DataFrame with predictions and ground truth
        coefficients: Model coefficients [B0, B1, B2]
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

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
    plt.savefig('classification_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
```

**Verification**:
- File classification_plot.png created
- Resolution ≥ 1200x800
- All elements visible (circles, X markers, decision boundary, legend)

---

### STEP 4.2: Training Convergence Visualization (FR-4.2)

**Task**: Create dual-axis plot showing log-likelihood and MSE over iterations

**Implementation Template**:
```python
def plot_training_convergence(training_history: List[Dict]):
    """Create training convergence visualization.

    Args:
        training_history: List of dicts with iteration, log_likelihood, mse
    """
    iterations = [h['iteration'] for h in training_history]
    log_likelihoods = [h['log_likelihood'] for h in training_history]
    mses = [h['mse'] for h in training_history]

    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=100)

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
    plt.savefig('training_convergence.png', dpi=100, bbox_inches='tight')
    plt.close()
```

**Verification**:
- File training_convergence.png created
- Both metrics visible
- Clear convergence pattern (metrics improving over iterations)

---

### STEP 4.3: Analysis Report (FR-4.3)

**Task**: Generate analysis.md with all required sections

**Key Sections** (from PRD):
1. Model Performance Summary
2. Training Convergence Analysis
3. Interesting Data Points
4. Sample Data Table
5. Model Coefficients
6. Insights

**Implementation Template**:
```python
def generate_analysis_report(
    predictions_df: pd.DataFrame,
    training_history: List[Dict],
    coefficients: np.ndarray
):
    """Generate comprehensive analysis report in markdown format.

    Args:
        predictions_df: Predictions with errors
        training_history: Training metrics history
        coefficients: Model coefficients [B0, B1, B2]
    """
    # Calculate statistics
    accuracy = (predictions_df['error_squared'] == 0).mean() * 100
    mse = predictions_df['error_squared'].mean()
    avg_error = np.sqrt(mse)
    final_log_likelihood = training_history[-1]['log_likelihood']
    n_iterations = len(training_history)
    n_misclassifications = (predictions_df['error_squared'] == 1).sum()

    # Find interesting points
    correct_predictions = predictions_df[predictions_df['error_squared'] == 0]
    incorrect_predictions = predictions_df[predictions_df['error_squared'] == 1]

    # Training convergence metrics
    initial_log_likelihood = training_history[0]['log_likelihood']
    initial_mse = training_history[0]['mse']
    final_mse = training_history[-1]['mse']

    # Build markdown content
    report = f"""# Sigmoid Binary Classification Analysis Report

## 1. Model Performance Summary

- **Overall Accuracy**: {accuracy:.2f}%
- **Mean Squared Error (MSE)**: {mse:.4f}
- **Average Error**: {avg_error:.4f}
- **Final Log-Likelihood**: {final_log_likelihood:.2f}
- **Training Iterations**: {n_iterations}
- **Misclassifications**: {n_misclassifications} out of {len(predictions_df)} points

## 2. Training Convergence Analysis

The model training {'converged successfully' if n_iterations < 1000 else 'reached maximum iterations'} after {n_iterations} iterations.

**Convergence Metrics:**
- Initial Log-Likelihood: {initial_log_likelihood:.2f}
- Final Log-Likelihood: {final_log_likelihood:.2f}
- Improvement: {final_log_likelihood - initial_log_likelihood:.2f}

- Initial MSE: {initial_mse:.4f}
- Final MSE: {final_mse:.4f}
- Reduction: {(initial_mse - final_mse):.4f}

**Convergence Behavior**: The model showed {'rapid' if n_iterations < 100 else 'steady' if n_iterations < 500 else 'slow'} convergence, with both log-likelihood and MSE {'improving consistently' if final_mse < initial_mse else 'showing instability'} throughout training.

See the training convergence visualization in `training_convergence.png` for detailed progression.

## 3. Interesting Data Points

"""

    # Most accurate (correct prediction)
    if len(correct_predictions) > 0:
        most_accurate = correct_predictions.iloc[0]
        report += f"""### Most Accurate Prediction (Correctly Classified)
- **Coordinates**: (x1={most_accurate['x1']:.4f}, x2={most_accurate['x2']:.4f})
- **Ground Truth (R)**: {int(most_accurate['R'])}
- **Prediction (σ)**: {int(most_accurate['sigma'])}
- **Squared Error**: {most_accurate['error_squared']:.0f} (Correct)
- **Continuous Output**: {most_accurate['sigma_continuous']:.4f}

"""

    # Least accurate (misclassified)
    if len(incorrect_predictions) > 0:
        least_accurate = incorrect_predictions.iloc[0]
        report += f"""### Least Accurate Prediction (Misclassified)
- **Coordinates**: (x1={least_accurate['x1']:.4f}, x2={least_accurate['x2']:.4f})
- **Ground Truth (R)**: {int(least_accurate['R'])}
- **Prediction (σ)**: {int(least_accurate['sigma'])}
- **Squared Error**: {least_accurate['error_squared']:.0f} (Incorrect)
- **Continuous Output**: {least_accurate['sigma_continuous']:.4f}

"""
    else:
        report += "**No Misclassifications**: All data points were correctly classified!\n\n"

    # Sample data table
    sample_df = predictions_df.head(15)
    report += "## 4. Sample Data Table\n\n"
    report += sample_df.to_markdown(index=False, floatfmt=".4f")
    report += "\n\n"

    # Model coefficients
    B0, B1, B2 = coefficients
    report += f"""## 5. Model Coefficients

The trained logistic regression model has the following coefficients:

- **B0 (Intercept)**: {B0:.4f}
- **B1 (x1 coefficient)**: {B1:.4f}
- **B2 (x2 coefficient)**: {B2:.4f}

**Decision Boundary Equation:**
```
σ(x1, x2) = 1 / (1 + exp(-(B0 + B1*x1 + B2*x2)))
Decision boundary: B0 + B1*x1 + B2*x2 = 0
Simplified: x2 = {-B0/B2:.4f} + {-B1/B2:.4f}*x1
```

The decision boundary is a straight line that separates the two classes in the 2D feature space.

## 6. Insights

"""

    # Generate insights based on results
    if accuracy >= 95:
        sep_quality = "excellent"
    elif accuracy >= 80:
        sep_quality = "good"
    else:
        sep_quality = "moderate"

    report += f"""### Classification Performance
- The model achieved {sep_quality} separation between the two clusters with {accuracy:.2f}% accuracy.
- {n_misclassifications} data points were misclassified, indicating {'minimal overlap' if n_misclassifications < 10 else 'some overlap'} between clusters.

### Model Behavior
- The {'negative' if B1 < 0 else 'positive'} B1 coefficient indicates that increasing x1 {'decreases' if B1 < 0 else 'increases'} the probability of class 1.
- The {'negative' if B2 < 0 else 'positive'} B2 coefficient indicates that increasing x2 {'decreases' if B2 < 0 else 'increases'} the probability of class 1.

### Cluster Separation
- The two clusters are {sep_quality}ly separated in the feature space.
- The linear decision boundary {'effectively' if accuracy >= 80 else 'partially'} captures the separation between classes.
- {'Perfect linear separation' if accuracy == 100 else 'The sigmoid function provides a smooth probabilistic boundary'} is achieved.

---

*This report was automatically generated by the Sigmoid Binary Classification System.*
"""

    # Save report
    with open('analysis.md', 'w') as f:
        f.write(report)
```

**Pitfalls to Avoid**:
- ❌ Don't skip any required sections
- ❌ Don't forget to handle case when no misclassifications exist
- ❌ Don't forget to format the sample table as markdown
- ❌ Don't hard-code values; calculate everything from data

**Verification**:
```python
generate_analysis_report(predictions_df, training_history, coefficients)
assert os.path.exists('analysis.md'), "analysis.md must be created"
with open('analysis.md', 'r') as f:
    content = f.read()
    assert '## 1. Model Performance Summary' in content
    assert '## 2. Training Convergence Analysis' in content
    assert '## 3. Interesting Data Points' in content
    assert '## 4. Sample Data Table' in content
    assert '## 5. Model Coefficients' in content
    assert '## 6. Insights' in content
print("✓ Step 4.3 complete: analysis.md created with all sections")
```

---

### STEP 5: Main Orchestration

**Task**: Create main.py to orchestrate all steps

**Implementation Template**:
```python
"""
Sigmoid Binary Classification System
Main execution script
"""

import numpy as np
import pandas as pd
from data_generator import generate_dataset
from model_trainer import train_logistic_regression
from analyzer import (
    predict_and_calculate_errors,
    plot_classification,
    plot_training_convergence,
    generate_analysis_report
)

# Configuration
CONFIG = {
    'n_samples_per_cluster': 100,
    'cluster_1_center': (0.3, 0.3),
    'cluster_2_center': (0.7, 0.7),
    'cluster_std': 0.1,
    'random_seed': 42,
    'learning_rate': 0.1,
    'max_iterations': 1000,
    'convergence_tolerance': 1e-6
}

def main():
    """Main execution function."""
    print("=" * 60)
    print("Sigmoid Binary Classification System")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\n[Step 1/4] Generating synthetic dataset...")
    dataset_df = generate_dataset(
        n_samples_per_cluster=CONFIG['n_samples_per_cluster'],
        cluster_1_center=CONFIG['cluster_1_center'],
        cluster_2_center=CONFIG['cluster_2_center'],
        cluster_std=CONFIG['cluster_std'],
        random_seed=CONFIG['random_seed']
    )
    dataset_df.to_csv('dataset.csv', index=False)
    print(f"✓ Dataset created: {len(dataset_df)} samples saved to dataset.csv")

    # Step 2: Train model
    print("\n[Step 2/4] Training logistic regression model...")
    X = dataset_df[['x1', 'x2']].values
    y = dataset_df['label'].values

    coefficients, training_history = train_logistic_regression(
        X, y,
        learning_rate=CONFIG['learning_rate'],
        max_iterations=CONFIG['max_iterations'],
        tolerance=CONFIG['convergence_tolerance']
    )
    print(f"✓ Model trained: {len(training_history)} iterations")
    print(f"  Coefficients: B0={coefficients[0]:.4f}, B1={coefficients[1]:.4f}, B2={coefficients[2]:.4f}")

    # Step 3: Generate predictions
    print("\n[Step 3/4] Generating predictions and calculating errors...")
    predictions_df = predict_and_calculate_errors(X, y, coefficients)
    predictions_df.to_csv('predictions.csv', index=False)
    accuracy = (predictions_df['error_squared'] == 0).mean() * 100
    print(f"✓ Predictions saved to predictions.csv")
    print(f"  Accuracy: {accuracy:.2f}%")

    # Step 4: Generate visualizations and analysis
    print("\n[Step 4/4] Creating visualizations and analysis report...")
    plot_classification(predictions_df, coefficients)
    print("  ✓ Classification plot saved to classification_plot.png")

    plot_training_convergence(training_history)
    print("  ✓ Training convergence plot saved to training_convergence.png")

    generate_analysis_report(predictions_df, training_history, coefficients)
    print("  ✓ Analysis report saved to analysis.md")

    print("\n" + "=" * 60)
    print("All steps completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - dataset.csv")
    print("  - predictions.csv")
    print("  - classification_plot.png")
    print("  - training_convergence.png")
    print("  - analysis.md")
    print("\nRun `cat analysis.md` to view the analysis report.")

if __name__ == "__main__":
    main()
```

**Verification**:
Run `uv run python main.py` and check:
- [ ] All 5 output files created
- [ ] Console output shows progress
- [ ] No errors or warnings (unless expected)
- [ ] Execution time < 10 seconds

---

## Code Quality Checklist

Before completing implementation:

### Type Hints
- [ ] All function signatures have type hints
- [ ] Use `from typing import List, Dict, Tuple` as needed
- [ ] np.ndarray for NumPy arrays
- [ ] pd.DataFrame for Pandas DataFrames

### Docstrings
- [ ] All public functions have Google-style docstrings
- [ ] Include: Brief description, Args, Returns, Raises (if applicable)

### Line Limits
- [ ] main.py ≤ 250 lines
- [ ] data_generator.py ≤ 250 lines
- [ ] model_trainer.py ≤ 250 lines
- [ ] analyzer.py ≤ 250 lines
- [ ] If any file exceeds 250 lines, split into smaller modules

### Error Handling
- [ ] Validate inputs where appropriate
- [ ] Handle convergence failures gracefully
- [ ] Provide helpful error messages
- [ ] Warn for edge cases (clipping, slow convergence)

---

## Testing Strategy for AI Implementation

### Unit Testing Approach

Create simple tests for each component:

```python
# Test data generation
def test_data_generation():
    df = generate_dataset(n_samples_per_cluster=50)
    assert df.shape == (100, 3)
    assert set(df['label'].unique()) == {0, 1}
    assert df['x1'].between(0, 1).all()
    assert df['x2'].between(0, 1).all()

# Test sigmoid function
def test_sigmoid():
    assert 0 < sigmoid(np.array([0]))[0] < 1
    assert sigmoid(np.array([0]))[0] == 0.5
    assert sigmoid(np.array([10]))[0] > 0.99
    assert sigmoid(np.array([-10]))[0] < 0.01

# Test error calculation
def test_error_calculation():
    # Perfect prediction
    assert ((1 - 1) ** 2) == 0
    # Wrong prediction
    assert ((0 - 1) ** 2) == 1
```

### Integration Testing

Test the full pipeline:
```python
def test_full_pipeline():
    # Run main
    main()

    # Check outputs exist
    assert os.path.exists('dataset.csv')
    assert os.path.exists('predictions.csv')
    assert os.path.exists('classification_plot.png')
    assert os.path.exists('training_convergence.png')
    assert os.path.exists('analysis.md')

    # Verify predictions
    pred_df = pd.read_csv('predictions.csv')
    assert set(pred_df['sigma'].unique()).issubset({0, 1})
    assert set(pred_df['error_squared'].unique()).issubset({0, 1})
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Numerical Instability
**Problem**: Overflow/underflow in exp() calculation
**Solution**: Clip z values before sigmoid: `z = np.clip(z, -500, 500)`

### Pitfall 2: Log(0) Error
**Problem**: log(0) when computing log-likelihood
**Solution**: Add epsilon: `y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)`

### Pitfall 3: Wrong Error Formula
**Problem**: Using continuous sigmoid for MSE instead of binary
**Solution**: Convert to binary first: `sigma = (sigma_continuous >= 0.5).astype(int)`

### Pitfall 4: Missing Bias Term
**Problem**: Forgetting to add column of ones for B0
**Solution**: `X_with_bias = np.column_stack([np.ones(n), X])`

### Pitfall 5: Forgetting Iteration Tracking
**Problem**: Not storing metrics at each iteration
**Solution**: Store in list within training loop, not just final values

### Pitfall 6: Incorrect Decision Boundary
**Problem**: Decision boundary doesn't match predictions
**Solution**: Ensure same coefficients and threshold (0.5) used

### Pitfall 7: Incomplete analysis.md
**Problem**: Missing required sections
**Solution**: Use checklist, verify all 6 sections present

---

## Final Verification Checklist

Before declaring implementation complete:

### Functional Requirements
- [ ] FR-1.1: dataset.csv created with correct columns
- [ ] FR-2.1: Model trained with iteration tracking
- [ ] FR-3.1: predictions.csv created with all columns
- [ ] FR-4.1: classification_plot.png created
- [ ] FR-4.2: training_convergence.png created
- [ ] FR-4.3: analysis.md with all 6 sections

### Output Validation
- [ ] dataset.csv: 200 rows (or 2×n_samples_per_cluster), 3 columns (x1, x2, label)
- [ ] predictions.csv: Same number of rows, 6 columns (x1, x2, R, sigma_continuous, sigma, error_squared)
- [ ] classification_plot.png: Valid image, ≥1200×800 pixels
- [ ] training_convergence.png: Valid image, shows both metrics
- [ ] analysis.md: Valid markdown with all required sections

### Code Quality
- [ ] All files ≤ 250 lines
- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] Configuration centralized
- [ ] Clean console output

### Performance
- [ ] Total execution time < 10 seconds
- [ ] Accuracy > 80% (with default parameters)
- [ ] Model converges (or reaches max iterations gracefully)

### Documentation
- [ ] README.md created with usage instructions
- [ ] Code comments where needed
- [ ] Clear error messages

---

## README.md Template

```markdown
# Sigmoid Binary Classification System

A Python-based machine learning system that generates synthetic 2D data and uses logistic regression with sigmoid activation for binary classification.

## Installation

### Prerequisites
- Python 3.8 or higher
- UV package manager

### Setup

\`\`\`bash
# Install UV if not already installed
pip install uv

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
uv pip install numpy pandas matplotlib
\`\`\`

## Usage

Run the complete classification pipeline:

\`\`\`bash
uv run python main.py
\`\`\`

## Configuration

Edit the `CONFIG` dictionary in `main.py` to adjust parameters:

- `n_samples_per_cluster`: Number of points per cluster (default: 100)
- `cluster_1_center`: Center coordinates for cluster 0 (default: (0.3, 0.3))
- `cluster_2_center`: Center coordinates for cluster 1 (default: (0.7, 0.7))
- `cluster_std`: Standard deviation for clusters (default: 0.1)
- `random_seed`: Random seed for reproducibility (default: 42)
- `learning_rate`: Gradient descent learning rate (default: 0.1)
- `max_iterations`: Maximum training iterations (default: 1000)
- `convergence_tolerance`: Convergence threshold (default: 1e-6)

## Output Files

The system generates 5 files:

1. **dataset.csv**: Generated synthetic data with features (x1, x2) and labels
2. **predictions.csv**: Model predictions and squared errors for each data point
3. **classification_plot.png**: Visualization of ground truth, predictions, and decision boundary
4. **training_convergence.png**: Training metrics (log-likelihood and MSE) over iterations
5. **analysis.md**: Comprehensive statistical analysis and insights

## Project Structure

- `main.py`: Main orchestration script
- `data_generator.py`: Synthetic data generation
- `model_trainer.py`: Logistic regression implementation
- `analyzer.py`: Prediction, visualization, and reporting

## Requirements

- NumPy >= 1.20.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Python >= 3.8

## License

MIT
\`\`\`

---

## Execution Instructions for Claude

When asked to implement this project:

1. **Read all documentation first**: PRD.md, PLANNING.md, TODO.md
2. **Follow this guide step-by-step**: Don't skip steps
3. **Verify after each step**: Use the verification code snippets
4. **Track progress**: Mark items in TODO.md as you complete them
5. **Test frequently**: Run tests after each major component
6. **Ask for clarification**: If requirements are unclear
7. **Use the templates**: Provided code templates are tested and work
8. **Check the pitfalls**: Review common pitfalls before implementing each step
9. **Validate outputs**: Ensure all 5 files are created correctly
10. **Final review**: Run the complete checklist before declaring done

**Remember**: The goal is a working, well-documented, maintainable system that meets all PRD requirements. Quality over speed!

---

**Document Version**: 1.0
**Created**: 2025-11-15
**Status**: Ready for Use
