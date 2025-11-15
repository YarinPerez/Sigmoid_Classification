"""
Analysis module for predictions and statistical reports.

This module handles predictions, error calculation, and generates
comprehensive markdown analysis reports.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from model_trainer import sigmoid


def predict_and_calculate_errors(
    X: np.ndarray,
    y_true: np.ndarray,
    coefficients: np.ndarray
) -> pd.DataFrame:
    """Generate predictions and calculate squared errors.

    Args:
        X: Feature matrix of shape (n_samples, 2)
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


def generate_analysis_report(
    predictions_df: pd.DataFrame,
    training_history: List[Dict],
    coefficients: np.ndarray,
    output_file: str = 'analysis.md'
) -> None:
    """Generate comprehensive analysis report in markdown format.

    Args:
        predictions_df: DataFrame with predictions and errors
        training_history: List of training metrics dictionaries
        coefficients: Model coefficients [B0, B1, B2]
        output_file: Output filename (default: 'analysis.md')
    """
    # Calculate statistics
    accuracy = (predictions_df['error_squared'] == 0).mean() * 100
    mse = predictions_df['error_squared'].mean()
    avg_error = np.sqrt(mse)
    final_log_likelihood = training_history[-1]['log_likelihood']
    n_iterations = len(training_history)
    n_misclassifications = int((predictions_df['error_squared'] == 1).sum())

    # Find interesting points
    correct_predictions = predictions_df[predictions_df['error_squared'] == 0]
    incorrect_predictions = predictions_df[predictions_df['error_squared'] == 1]

    # Training convergence metrics
    initial_log_likelihood = training_history[0]['log_likelihood']
    initial_mse = training_history[0]['mse']
    final_mse = training_history[-1]['mse']

    # Build markdown content
    report = f"""# Sigmoid Binary Classification Analysis Report

## Data Files & Visualizations

### Input Data
- **[dataset.csv](dataset.csv)** - Original synthetic training data (200 samples, 3 columns: x1, x2, label)
- **[predictions.csv](predictions.csv)** - Model predictions with errors (200 rows, 6 columns: x1, x2, R, sigma_continuous, sigma, error_squared)

### Generated Visualizations

#### Classification Visualization
![Classification Plot](classification_plot.png)

*Figure 1: Binary classification results showing ground truth (circles) vs predictions (X markers) with decision boundary (green line)*

#### Training Convergence
![Training Convergence](training_convergence.png)

*Figure 2: Training convergence showing log-likelihood improvement and MSE reduction over iterations*

---

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

**Convergence Behavior**: The model showed {'rapid' if n_iterations < 100 else 'steady' if n_iterations < 500 else 'slow'} convergence, with both metrics {'improving consistently' if final_mse < initial_mse else 'showing instability'} throughout training.

See the training convergence visualization in `training_convergence.png` for detailed progression.

## 3. Interesting Data Points

"""

    report += "### Correct Datapoint (Correctly Classified)\n\n"

    # One correct prediction
    if len(correct_predictions) > 0:
        correct_point = correct_predictions.iloc[0]
        report += f"""- **Coordinates**: (x1={correct_point['x1']:.4f}, x2={correct_point['x2']:.4f})
- **Ground Truth (R)**: {int(correct_point['R'])}
- **Prediction (σ)**: {int(correct_point['sigma'])}
- **Continuous Sigmoid Output**: {correct_point['sigma_continuous']:.4f}
- **Squared Error**: {correct_point['error_squared']:.0f} ✓ (Correct Classification)

"""
    else:
        report += "No correct predictions found.\n\n"

    report += "### Incorrect Datapoint (Misclassified)\n\n"

    # One incorrect prediction
    if len(incorrect_predictions) > 0:
        incorrect_point = incorrect_predictions.iloc[0]
        report += f"""- **Coordinates**: (x1={incorrect_point['x1']:.4f}, x2={incorrect_point['x2']:.4f})
- **Ground Truth (R)**: {int(incorrect_point['R'])}
- **Prediction (σ)**: {int(incorrect_point['sigma'])}
- **Continuous Sigmoid Output**: {incorrect_point['sigma_continuous']:.4f}
- **Squared Error**: {incorrect_point['error_squared']:.0f} ✗ (Incorrect Classification)

"""
    else:
        report += "**Perfect Classification**: No misclassifications found! All 200 data points were correctly classified. ✓\n\n"

    # Sample data table
    sample_df = predictions_df.head(15)
    report += "## 4. Sample Data Table\n\n"
    report += "**Data Source**: [predictions.csv](predictions.csv) (showing first 15 rows)\n\n"
    report += sample_df.to_markdown(index=False, floatfmt=".4f")
    report += "\n\n**Full Dataset**: For the complete dataset with all 200 predictions, see [predictions.csv](predictions.csv)\n\n"

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
```

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

---

*This report was automatically generated by the Sigmoid Binary Classification System.*
"""

    # Save report with UTF-8 encoding to support Unicode characters
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
