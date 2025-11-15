# Product Requirements Document: Sigmoid Binary Classification System

## 1. Overview

### 1.1 Product Name
Sigmoid Binary Classifier

### 1.2 Purpose
A Python-based machine learning program that generates synthetic 2D clustered data and uses logistic regression with sigmoid activation to classify the two clusters. The system provides comprehensive analysis and visualization of classification performance.

### 1.3 Target Users
- Data scientists exploring binary classification
- Students learning logistic regression concepts
- ML practitioners prototyping classification algorithms

---

## 2. Technical Requirements

### 2.1 Environment
- **Runtime**: Python 3.8+
- **Package Manager**: UV (Universal Virtualenv)
- **Virtual Environment**: Must run under `uv venv`

### 2.2 Dependencies
- NumPy (numerical operations)
- Pandas (data manipulation)
- Matplotlib (visualization)
- Scikit-learn or statsmodels (logistic regression)

---

## 3. Functional Requirements

### 3.1 Step 1: Dataset Generation

**Requirement ID**: FR-1.1

**Description**: Generate a synthetic 2D dataset with two distinct clusters.

**Specifications**:
- Create two clusters of data points
- Each point has two features: x1 (x-coordinate) and x2 (y-coordinate)
- Data range: [0, 1] for both dimensions
- Random generation with controllable parameters (cluster centers, spread, sample size)
- Binary labels (R): 0 for cluster 1, 1 for cluster 2

**Output**:
- File: `dataset.csv`
- Columns: `x1`, `x2`, `label` (where label is 0 or 1)
- Format: CSV with headers

**Acceptance Criteria**:
- ✓ Two visually separable clusters
- ✓ All values within [0, 1] range
- ✓ Balanced or near-balanced class distribution
- ✓ Successfully saved to file

---

### 3.2 Step 2: Logistic Regression Model Training

**Requirement ID**: FR-2.1

**Description**: Train a logistic regression model to find optimal coefficients for sigmoid classification, tracking the training progress.

**Mathematical Model**:
```
Sigmoid function: σ(z) = 1 / (1 + e^(-z))
where z = B0 + B1·x1 + B2·x2

Logit transformation: logit(p) = log(p / (1-p)) = B0 + B1·x1 + B2·x2
```

**Specifications**:
- Use logistic regression to estimate coefficients B0, B1, B2
- Fit model on generated dataset
- Store learned parameters for prediction
- **Track training progress at each iteration/epoch**:
  - **Log-likelihood**: Sum of log-likelihoods across all data points
    - Formula: `Σ[R*log(σ_continuous) + (1-R)*log(1-σ_continuous)]`
    - Where σ_continuous = 1 / (1 + e^(-(B0 + B1·x1 + B2·x2)))
  - **Mean Squared Error**: Average of (σ - R)² for all data points (after binary classification)

**Output**:
- Model coefficients: B0 (intercept), B1 (x1 coefficient), B2 (x2 coefficient)
- Model object for predictions
- **Training history**: List of (iteration, log-likelihood, mean_squared_error) tuples

**Implementation Notes**:
- **Recommended**: Use custom gradient descent implementation to track iteration history
  - Learning rate: 0.1 (configurable)
  - Max iterations: 1000 (configurable)
  - Convergence tolerance: 1e-6
- **Alternative**: If using sklearn's LogisticRegression, implement a callback or iterative training loop to capture metrics at each step
- Must be able to extract and store metrics at each iteration for convergence visualization

**Acceptance Criteria**:
- ✓ Successfully converged regression model
- ✓ Three coefficients computed
- ✓ Model can generate predictions
- ✓ Training metrics captured at each step

---

### 3.3 Step 3: Prediction and Error Calculation

**Requirement ID**: FR-3.1

**Description**: Generate predictions for all data points and calculate squared errors, with both continuous sigmoid output and binary classification.

**Specifications**:
- For each data point (x1, x2):
  - Calculate continuous sigmoid: σ_continuous(x1, x2) = 1 / (1 + e^(-(B0 + B1·x1 + B2·x2)))
  - Convert to binary classification: σ(x1, x2) = 1 if σ_continuous ≥ 0.5, else 0
  - Compare with ground truth label R (which is 0 or 1)
  - Compute squared error: (σ - R)²
    - This will be 0 if classification is correct (σ = R)
    - This will be 1 if classification is incorrect (σ ≠ R)

**Output**:
- File: `predictions.csv`
- Columns: 
  - `x1`, `x2`: Input features
  - `R`: Ground truth (0 or 1)
  - `sigma_continuous`: Continuous sigmoid output [0.0, 1.0] (optional but recommended)
  - `sigma`: Binary predicted class (0 or 1)
  - `error_squared`: (σ - R)²
- Format: CSV with headers

**Acceptance Criteria**:
- ✓ Binary predictions (0 or 1) for all data points
- ✓ R values are 0 or 1 (cluster labels)
- ✓ Sigma values are 0 or 1 (predicted cluster labels)
- ✓ Squared error correctly calculated as (σ - R)²
- ✓ Successfully saved to file

---

### 3.4 Step 4: Analysis and Visualization

**Requirement ID**: FR-4.1

**Description**: Create comprehensive visualization and statistical analysis of classification results.

#### 3.4.1 Classification Visualization

**Specifications**:
- Generate 2D scatter plot with:
  - Ground truth labels: circles (○) with different colors for each cluster (0 vs 1)
  - Predictions: X markers colored by predicted cluster (0 vs 1)
  - Decision boundary: sigmoid contour line (where σ_continuous = 0.5)
- Axes: x1 (horizontal), x2 (vertical)
- Range: [0, 1] for both axes
- Legend clearly identifying:
  - Cluster 0 (truth) - circles
  - Cluster 1 (truth) - circles
  - Predicted Cluster 0 - X markers
  - Predicted Cluster 1 - X markers
  - Decision boundary line

**Output**:
- File: `classification_plot.png`
- Format: PNG image
- Resolution: Minimum 1200x800 pixels
- DPI: 100+

**Acceptance Criteria**:
- ✓ All data points visible
- ✓ Clear distinction between truth (circles) and predictions (X markers)
- ✓ Colors match cluster assignments (0 or 1)
- ✓ Decision boundary shown where classification switches from 0 to 1
- ✓ Professional appearance with labels and legend

---

#### 3.4.2 Training Convergence Visualization

**Requirement ID**: FR-4.2

**Description**: Visualize the regression training process showing convergence of likelihood and error over iterations.

**Specifications**:
- Generate dual-axis line plot with:
  - X-axis: Iteration/step number
  - Left Y-axis: Log-likelihood value (logit(p) = B0 + B1·x1 + B2·x2)
  - Right Y-axis: Mean squared error (average of (σ - R)²)
  - Two lines showing progression of both metrics
- Different colors for each metric
- Grid for readability
- Legend identifying each line

**Output**:
- File: `training_convergence.png`
- Format: PNG image
- Resolution: Minimum 1200x800 pixels
- DPI: 100+

**Acceptance Criteria**:
- ✓ Both metrics plotted over all iterations
- ✓ Clear convergence pattern visible
- ✓ Dual y-axes properly scaled and labeled
- ✓ Professional appearance with title, labels, and legend

---

#### 3.4.3 Analysis Report

**Requirement ID**: FR-4.3

**Description**: Generate markdown report with statistical analysis and insights.

**Output**:
- File: `analysis.md`

**Required Sections**:

1. **Model Performance Summary**
   - Overall accuracy (percentage of correct classifications where σ = R)
   - Mean squared error (MSE) - average of all (σ - R)² values
   - Average error across all data points
   - Final log-likelihood value
   - Number of training iterations until convergence
   - Number of misclassifications (where σ ≠ R)

2. **Training Convergence Analysis**
   - Description of convergence behavior
   - Initial vs final log-likelihood
   - Initial vs final MSE
   - Rate of convergence (fast/slow/unstable)
   - Reference to training_convergence.png graph

3. **Interesting Data Points**
   - Most accurate prediction (error_squared = 0, correctly classified)
     - Coordinates (x1, x2)
     - Ground truth (R: 0 or 1) vs prediction (σ: 0 or 1)
     - Error value (0)
     - Note: If all correct, pick one example
   - Least accurate prediction (error_squared = 1, misclassified)
     - Coordinates (x1, x2)
     - Ground truth (R: 0 or 1) vs prediction (σ: 0 or 1)
     - Error value (1)
     - Note: If all correct, state "No misclassifications"

4. **Sample Data Table**
   - Display 10-15 sample rows from predictions table
   - Formatted as markdown table
   - Include all columns

5. **Model Coefficients**
   - B0 (intercept)
   - B1 (x1 coefficient)
   - B2 (x2 coefficient)
   - Interpretation of decision boundary

6. **Insights**
   - Classification patterns observed
   - Model strengths/weaknesses
   - Cluster separation quality

**Acceptance Criteria**:
- ✓ All required sections present
- ✓ Correct statistical calculations
- ✓ Formatted as valid markdown
- ✓ Professional and informative

---

## 4. Non-Functional Requirements

### 4.1 Performance
- Dataset generation: < 1 second for up to 1000 points
- Model training: < 5 seconds
- Visualization generation: < 3 seconds
- Total execution time: < 10 seconds

### 4.2 Reliability
- Deterministic results with seed control
- Graceful error handling
- Input validation

### 4.3 Usability
- Single command execution: `uv run python main.py`
- Clear console output showing progress
- All outputs automatically saved to files

### 4.4 Maintainability
- Modular code structure (separate functions for each step)
- Type hints for functions
- Docstrings for all major functions
- Configuration parameters at top of file or in config file
- **Source code files limited to maximum 250 lines each**
- If main.py exceeds 250 lines, split into multiple modules

---

## 5. File Structure

```
project/
├── main.py                    # Main execution script (orchestrates steps)
├── data_generator.py          # Step 1: Dataset generation (optional if main.py > 250 lines)
├── model_trainer.py           # Step 2: Logistic regression (optional if main.py > 250 lines)
├── analyzer.py                # Step 3-4: Prediction & analysis (optional if main.py > 250 lines)
├── pyproject.toml            # UV/pip dependencies
├── README.md                 # Usage instructions (see section 5.1 below)
├── dataset.csv               # Generated (Step 1 output)
├── predictions.csv           # Generated (Step 3 output)
├── classification_plot.png   # Generated (Step 4.1 output)
├── training_convergence.png  # Generated (Step 4.2 output)
└── analysis.md               # Generated (Step 4.3 output)
```

**Note**: Source code files must not exceed 250 lines. If implementation requires more, split functionality across multiple modules as shown above.

### 5.1 README.md Requirements

The README.md file should contain:

1. **Project Title and Description**
   - Brief overview of the sigmoid binary classifier

2. **Installation**
   ```bash
   # Clone or download the project
   # Install UV if not already installed
   pip install uv
   
   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. **Usage**
   ```bash
   uv run python main.py
   ```

4. **Configuration**
   - List of configurable parameters and their defaults
   - How to modify them (if applicable)

5. **Output Files**
   - Description of each generated file
   - Where to find them

6. **Requirements**
   - Python version
   - Dependencies list

---

## 6. Configuration Parameters

The following parameters should be configurable (hardcoded or via config):

- `n_samples_per_cluster`: Number of points per cluster (default: 100, total dataset = 200 points)
- `cluster_1_center`: (x1, x2) center for cluster 1 (default: (0.3, 0.3))
- `cluster_2_center`: (x1, x2) center for cluster 2 (default: (0.7, 0.7))
- `cluster_std`: Standard deviation for clusters (default: 0.1)
- `random_seed`: Seed for reproducibility (default: 42)
- `learning_rate`: Learning rate for gradient descent (default: 0.1)
- `max_iterations`: Maximum training iterations (default: 1000)
- `convergence_tolerance`: Convergence threshold (default: 1e-6)

---

## 7. Success Metrics

- **Functional Completeness**: All 4 steps execute successfully
- **Classification Performance**: Accuracy > 80% on generated data
- **File Generation**: All 5 output files created correctly (dataset.csv, predictions.csv, classification_plot.png, training_convergence.png, analysis.md)
- **Visual Quality**: Classification plot clearly shows clusters and decision boundary
- **Training Visualization**: Convergence graph shows clear progression of log-likelihood and MSE
- **Report Quality**: Analysis.md contains all required sections with accurate data including convergence analysis

---

---

## 7.5 Error Handling Requirements

### 7.5.1 Data Generation Errors
- Validate that all generated points fall within [0, 1] range
- If cluster centers + spread would generate out-of-range points, clip to [0, 1]
- Log warning if significant clipping occurs (> 5% of points)

### 7.5.2 Training Errors
- If model fails to converge within max_iterations:
  - Save partial results
  - Log warning with current loss value
  - Continue to prediction step with best-available coefficients
- Handle numerical instability (overflow/underflow in sigmoid):
  - Clip extreme values in sigmoid calculation
  - Use numerically stable implementations

### 7.5.3 Edge Cases
- **Perfect separation**: If clusters are perfectly separable, model should still complete successfully
- **Overlapping clusters**: If clusters overlap significantly, report lower accuracy but complete successfully
- **Degenerate cases**: Handle cases where all points classified to one cluster

### 7.5.4 File I/O Errors
- Check write permissions before saving files
- Provide clear error messages if files cannot be written
- Don't overwrite existing files without confirmation (or use timestamp in filename)

---

## 8. Constraints and Assumptions

### 8.1 Constraints
- Must use UV for environment management
- Data must be normalized to [0, 1] range
- Must use sigmoid function for classification
- Must use logistic regression (not other algorithms)
- **Source code files must not exceed 250 lines each**

### 8.2 Assumptions
- Clusters are linearly separable or near-linearly separable
- User has UV installed
- Python 3.8+ is available
- Sufficient disk space for output files

---

## 9. Testing Requirements

### 9.1 Unit Tests
- Data generation produces correct range
- Sigmoid function calculation is accurate
- Error calculation is correct
- File I/O operations succeed

### 9.2 Integration Tests
- Full pipeline runs end-to-end
- All output files created (dataset.csv, predictions.csv, classification_plot.png, training_convergence.png, analysis.md)
- Analysis.md contains expected sections including convergence analysis
- Both plot files are valid images
- Training convergence shows decreasing error over iterations

### 9.3 Validation Tests
- Coefficients produce reasonable decision boundary
- Accuracy metrics match manual calculation
- Most/least accurate points correctly identified

---

## 10. Future Enhancements (Out of Scope)

- Support for more than 2 clusters
- Interactive visualization
- Command-line arguments for parameters
- Cross-validation
- Additional classification metrics (precision, recall, F1)
- Support for non-linear decision boundaries
- Real-time training visualization

---

## 11. Glossary

- **Sigmoid**: S-shaped activation function mapping real numbers to (0, 1). For classification, values ≥ 0.5 are classified as 1, values < 0.5 as 0
- **Logit**: Inverse of sigmoid, log-odds transformation
- **Logistic Regression**: Statistical method for binary classification
- **Decision Boundary**: Line/curve separating the two classes (where sigmoid = 0.5)
- **MSE**: Mean Squared Error, average of squared prediction errors
- **UV**: Universal Virtualenv, modern Python package manager
- **R**: Ground truth label (0 or 1) indicating which cluster a point belongs to
- **σ (sigma)**: Predicted classification (0 or 1) based on thresholding the sigmoid output at 0.5
- **Binary Classification**: Assigning each data point to one of two classes (0 or 1)

---

## 12. Approval

This PRD defines the complete scope for the Sigmoid Binary Classification System.

**Document Version**: 1.0  
**Last Updated**: 2025-11-15  
**Status**: Ready for Implementation