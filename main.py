"""
Sigmoid Binary Classification System - Main Orchestration Script

A complete pipeline for binary classification using logistic regression
with sigmoid activation function.
"""

import numpy as np
import pandas as pd
from data_generator import generate_dataset
from model_trainer import train_logistic_regression
from analyzer import (
    predict_and_calculate_errors,
    generate_analysis_report
)
from visualizer import (
    plot_classification,
    plot_training_convergence
)

# Configuration Parameters
CONFIG = {
    'n_samples_per_cluster': 100,
    'cluster_1_center': (0.7, 0.3),
    'cluster_2_center': (0.3, 0.7),
    'cluster_std': 0.1,
    'random_seed': 42,
    'learning_rate': 0.1,
    'max_iterations': 1000,
    'convergence_tolerance': 1e-6
}


def main() -> None:
    """Main execution function that orchestrates all pipeline steps."""
    print("=" * 70)
    print("Sigmoid Binary Classification System")
    print("=" * 70)

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
    print(f"✓ Dataset created: {len(dataset_df)} samples")
    print(f"  Class 0: {(dataset_df['label'] == 0).sum()} samples")
    print(f"  Class 1: {(dataset_df['label'] == 1).sum()} samples")
    print(f"  Saved to: dataset.csv")

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
    print(f"✓ Model trained successfully")
    print(f"  Iterations: {len(training_history)}")
    print(f"  Coefficients:")
    print(f"    B0 (intercept): {coefficients[0]:.6f}")
    print(f"    B1 (x1 weight): {coefficients[1]:.6f}")
    print(f"    B2 (x2 weight): {coefficients[2]:.6f}")
    print(f"  Final log-likelihood: {training_history[-1]['log_likelihood']:.4f}")
    print(f"  Final MSE: {training_history[-1]['mse']:.6f}")

    # Step 3: Generate predictions
    print("\n[Step 3/4] Generating predictions and calculating errors...")
    predictions_df = predict_and_calculate_errors(X, y, coefficients)
    predictions_df.to_csv('predictions.csv', index=False)
    accuracy = (predictions_df['error_squared'] == 0).mean() * 100
    n_misclassified = (predictions_df['error_squared'] == 1).sum()
    print(f"✓ Predictions generated")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Misclassifications: {n_misclassified}/{len(predictions_df)}")
    print(f"  Saved to: predictions.csv")

    # Step 4: Generate visualizations and analysis
    print("\n[Step 4/4] Creating visualizations and analysis report...")
    plot_classification(predictions_df, coefficients)
    print(f"  ✓ Classification plot saved to: classification_plot.png")

    plot_training_convergence(training_history)
    print(f"  ✓ Training convergence plot saved to: training_convergence.png")

    generate_analysis_report(predictions_df, training_history, coefficients)
    print(f"  ✓ Analysis report saved to: analysis.md")

    print("\n" + "=" * 70)
    print("All steps completed successfully!")
    print("=" * 70)
    print("\nGenerated Output Files:")
    print("  1. dataset.csv - Generated synthetic data")
    print("  2. predictions.csv - Model predictions and errors")
    print("  3. classification_plot.png - Ground truth vs predictions visualization")
    print("  4. training_convergence.png - Training metrics over iterations")
    print("  5. analysis.md - Comprehensive analysis report")
    print("\nTo view the analysis report:")
    print("  cat analysis.md")


if __name__ == "__main__":
    main()
