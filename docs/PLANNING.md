# Planning Document: Sigmoid Binary Classification System

## 1. Executive Summary

This document outlines the implementation strategy for a Python-based binary classification system using logistic regression with sigmoid activation. The system will generate synthetic 2D data, train a classifier, and provide comprehensive visualizations and analysis.

**Key Objectives:**
- Implement all 4 functional steps (data generation, training, prediction, analysis)
- Achieve >80% classification accuracy
- Generate 5 output files (2 CSV, 2 PNG, 1 MD)
- Maintain code quality with files ≤250 lines
- Complete execution in <10 seconds

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                    (Orchestration Layer)                     │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐      ┌──────────────┐
│    Data      │    │    Model     │      │   Analyzer   │
│  Generator   │───▶│   Trainer    │─────▶│              │
│              │    │              │      │              │
└──────────────┘    └──────────────┘      └──────────────┘
        │                   │                     │
        ▼                   ▼                     ▼
  dataset.csv      training_history      predictions.csv
                                         classification_plot.png
                                         training_convergence.png
                                         analysis.md
```

### 2.2 Module Breakdown

#### Option A: Monolithic (if total ≤250 lines)
- **main.py**: Contains all functionality

#### Option B: Modular (recommended for maintainability)
- **main.py** (~50 lines): Orchestration, configuration, execution flow
- **data_generator.py** (~80 lines): Dataset generation logic
- **model_trainer.py** (~100 lines): Logistic regression implementation
- **analyzer.py** (~150 lines): Prediction, visualization, report generation
- **utils.py** (optional, ~50 lines): Shared utilities, validation functions

**Decision**: Start with Option B for better maintainability and easier testing.

---

## 3. Technical Decisions

### 3.1 Logistic Regression Implementation

**Decision: Custom Gradient Descent Implementation**

**Rationale:**
- PRD requires tracking metrics at each iteration
- sklearn's LogisticRegression doesn't expose iteration-level metrics easily
- Custom implementation provides full control over training loop
- Educational value for understanding the algorithm

**Alternative Considered:**
- sklearn with warm_start and iterative training
- Pros: Proven, optimized
- Cons: Complex to extract iteration metrics, less control

**Implementation Approach:**
```python
def train_logistic_regression(X, y, learning_rate=0.1, max_iterations=1000, tolerance=1e-6):
    # Initialize coefficients
    # Loop:
    #   - Calculate predictions
    #   - Compute log-likelihood
    #   - Compute MSE
    #   - Calculate gradients
    #   - Update coefficients
    #   - Check convergence
    #   - Store metrics
    # Return: coefficients, training_history
```

### 3.2 Data Generation Strategy

**Decision: NumPy's Normal Distribution with Clipping**

**Implementation:**
```python
cluster_1 = np.random.normal(loc=center_1, scale=std, size=(n, 2))
cluster_2 = np.random.normal(loc=center_2, scale=std, size=(n, 2))
# Clip to [0, 1]
data = np.clip(np.vstack([cluster_1, cluster_2]), 0, 1)
```

**Edge Case Handling:**
- Monitor clipping percentage
- Warn if >5% of points clipped
- Alternative: Truncated normal distribution (more complex)

### 3.3 Visualization Approach

**Library: Matplotlib**

**Classification Plot Design:**
- Scatter plot with two marker types (circles for truth, X for predictions)
- 4 colors needed: 2 for truth classes, 2 for predicted classes
- Decision boundary: Contour line at σ_continuous = 0.5
- Figure size: (12, 8) inches at 100 DPI = 1200x800 pixels

**Training Convergence Plot Design:**
- Dual Y-axis line plot
- Left axis: Log-likelihood (typically negative)
- Right axis: MSE (0 to 1 for binary)
- Different colors and line styles for clarity

### 3.4 Error Handling Strategy

**Tiered Approach:**

1. **Validation Errors** (prevent bad input):
   - Validate configuration parameters
   - Raise ValueError with clear messages

2. **Runtime Warnings** (non-fatal issues):
   - Clipping >5% of data points
   - Slow convergence
   - Log warnings but continue

3. **Graceful Degradation** (handle failures):
   - If training doesn't converge, use best coefficients found
   - If file write fails, print error but don't crash
   - Missing data: skip that analysis section

---

## 4. Data Flow

### 4.1 Step-by-Step Data Transformation

```
Step 1: Data Generation
├─ Input: Configuration parameters
├─ Process: Generate 2 Gaussian clusters
├─ Transform: Clip to [0, 1], assign labels
└─ Output: DataFrame(x1, x2, label) → dataset.csv

Step 2: Model Training
├─ Input: dataset.csv
├─ Process: Gradient descent optimization
├─ Track: Log-likelihood, MSE per iteration
└─ Output: Coefficients (B0, B1, B2), training_history

Step 3: Prediction
├─ Input: dataset.csv + coefficients
├─ Process: Apply sigmoid, threshold at 0.5
├─ Calculate: Squared errors (σ - R)²
└─ Output: DataFrame(x1, x2, R, σ_continuous, σ, error_squared) → predictions.csv

Step 4: Analysis
├─ Input: predictions.csv + training_history + coefficients
├─ Process:
│  ├─ Generate classification scatter plot
│  ├─ Generate training convergence plot
│  └─ Calculate statistics
└─ Output:
   ├─ classification_plot.png
   ├─ training_convergence.png
   └─ analysis.md
```

### 4.2 Key Data Structures

**Configuration Dictionary:**
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

**Training History Structure:**
```python
training_history = [
    {'iteration': 0, 'log_likelihood': -138.63, 'mse': 0.25},
    {'iteration': 1, 'log_likelihood': -120.45, 'mse': 0.22},
    # ...
]
```

---

## 5. Implementation Phases

### Phase 1: Foundation (Day 1, ~2 hours)
- Set up project structure
- Create pyproject.toml with dependencies
- Set up UV environment
- Create basic main.py skeleton
- Create configuration dictionary

**Deliverables:**
- Working UV environment
- Runnable (but empty) main.py
- Configuration centralized

### Phase 2: Data Generation (Day 1, ~1 hour)
- Implement data_generator.py
- Generate two clusters
- Add clipping and validation
- Save to CSV
- Unit test data generation

**Deliverables:**
- dataset.csv generated successfully
- Data validated to be in [0, 1]

### Phase 3: Model Training (Day 1-2, ~3 hours)
- Implement sigmoid function
- Implement gradient descent
- Add iteration tracking
- Calculate log-likelihood and MSE
- Test convergence
- Handle edge cases

**Deliverables:**
- Trained model coefficients
- Training history with metrics
- Convergence within max_iterations

### Phase 4: Prediction (Day 2, ~1 hour)
- Implement prediction function
- Calculate continuous and binary outputs
- Calculate squared errors
- Save to CSV

**Deliverables:**
- predictions.csv with all required columns

### Phase 5: Visualization (Day 2, ~2 hours)
- Create classification plot
- Create training convergence plot
- Ensure plots meet resolution requirements
- Test plot clarity

**Deliverables:**
- classification_plot.png
- training_convergence.png

### Phase 6: Analysis Report (Day 2, ~2 hours)
- Calculate all statistics
- Find most/least accurate points
- Generate markdown sections
- Format tables
- Save to analysis.md

**Deliverables:**
- Complete analysis.md with all sections

### Phase 7: Integration & Testing (Day 3, ~3 hours)
- Integrate all modules in main.py
- Add progress messages
- Error handling
- End-to-end testing
- Performance testing
- Create README.md

**Deliverables:**
- Fully functional system
- README.md
- All tests passing

### Phase 8: Polish & Documentation (Day 3, ~1 hour)
- Code review
- Add/improve docstrings
- Final testing with different seeds
- Verify all requirements met

**Deliverables:**
- Production-ready code
- Complete documentation

**Total Estimated Time: ~15 hours** (can be compressed to 2-3 days)

---

## 6. Risk Assessment

### 6.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Training doesn't converge | High | Medium | Use tested learning rate, add max_iterations limit, save best coefficients |
| Numerical instability in sigmoid | Medium | Low | Clip extreme values, use stable log implementations |
| Poor cluster separation | Medium | Low | Use well-separated default centers, allow configuration |
| File I/O failures | Low | Low | Add error handling, check permissions |
| Performance >10 seconds | Medium | Low | Profile code, optimize bottlenecks, use vectorized operations |

### 6.2 Implementation Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Code exceeds 250 lines per file | Medium | Medium | Plan modular structure upfront, split early |
| Missing required sections in analysis.md | Medium | Low | Use checklist, validate output |
| Plots not meeting resolution requirements | Low | Low | Test early, set figsize and DPI correctly |
| Dependency conflicts | Medium | Low | Use UV for isolation, pin versions |

---

## 7. Testing Strategy

### 7.1 Unit Testing Approach

**Test Coverage:**
- Data generation functions
- Sigmoid calculation
- Gradient descent step
- Error calculation
- File I/O operations

**Testing Tools:**
- pytest for test framework
- numpy.testing for numerical comparisons
- Mock file I/O for testing saves

### 7.2 Integration Testing

**Test Scenarios:**
1. **Happy Path**: Well-separated clusters, clean convergence
2. **Overlapping Clusters**: Test with higher std deviation
3. **Perfect Separation**: Clusters very far apart
4. **Edge Cases**: Single cluster center, identical centers

### 7.3 Validation Testing

**Manual Checks:**
- Visual inspection of plots
- Sample calculation of a few predictions
- Verify decision boundary makes sense
- Check analysis.md formatting

---

## 8. Performance Optimization

### 8.1 Optimization Strategies

1. **Vectorization**: Use NumPy operations instead of loops
2. **Efficient Libraries**: Leverage optimized NumPy/Pandas routines
3. **Early Stopping**: Stop training when converged
4. **Lazy Evaluation**: Only calculate what's needed

### 8.2 Expected Performance

```
Data Generation (200 points):   ~0.01 seconds
Model Training (1000 iters):    ~0.5 seconds
Predictions (200 points):       ~0.01 seconds
Visualization (2 plots):        ~1 second
Analysis Report:                ~0.1 seconds
Total:                          ~2 seconds (well under 10s limit)
```

---

## 9. Code Quality Standards

### 9.1 Coding Conventions

- **Style**: PEP 8 compliance
- **Type Hints**: All function signatures
- **Docstrings**: Google style for all public functions
- **Naming**: Descriptive variable names
- **Comments**: Explain "why", not "what"

### 9.2 Documentation Requirements

**Function Docstring Template:**
```python
def function_name(param1: type, param2: type) -> return_type:
    """Brief description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ErrorType: When error occurs
    """
```

### 9.3 Line Limit Strategy

**If approaching 250 lines:**
1. Extract helper functions to utils.py
2. Split visualization into separate module
3. Move configuration to separate file
4. Consider splitting analyzer.py into visualizer.py and reporter.py

---

## 10. Dependencies Management

### 10.1 Core Dependencies

```toml
[project]
name = "sigmoid-classifier"
version = "1.0.0"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "matplotlib>=3.4.0",
    "scikit-learn>=1.0.0",  # Optional: for comparison
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
]
```

### 10.2 Alternative Implementations

**For Logistic Regression:**
- **Primary**: Custom gradient descent (recommended)
- **Alternative 1**: statsmodels.api.Logit (more statistical output)
- **Alternative 2**: sklearn.linear_model.LogisticRegression (simplest)

**Decision**: Use custom implementation for iteration tracking requirement.

---

## 11. File Organization

### 11.1 Final Directory Structure

```
Sigmoid_Classification/
├── main.py                    # ~50 lines: Main orchestration
├── data_generator.py          # ~80 lines: Dataset generation
├── model_trainer.py           # ~100 lines: Logistic regression
├── analyzer.py                # ~150 lines: Prediction & analysis
├── utils.py                   # ~50 lines: Shared utilities (optional)
├── pyproject.toml            # Dependencies
├── README.md                 # User documentation
├── .venv/                    # Virtual environment (UV)
├── docs/
│   ├── PRD.md               # Product requirements (provided)
│   ├── TODO.md              # Task checklist
│   ├── PLANNING.md          # This file
│   └── CLAUDE.md            # AI implementation guide
├── tests/                    # Unit tests (optional)
│   ├── test_data_generator.py
│   ├── test_model_trainer.py
│   └── test_analyzer.py
└── outputs/                  # Generated files (optional subdirectory)
    ├── dataset.csv
    ├── predictions.csv
    ├── classification_plot.png
    ├── training_convergence.png
    └── analysis.md
```

**Note**: Outputs can be in root directory or outputs/ subdirectory.

---

## 12. Success Metrics

### 12.1 Functional Completeness

- [ ] Step 1 (Data Generation): dataset.csv created
- [ ] Step 2 (Training): Model converges, coefficients obtained
- [ ] Step 3 (Prediction): predictions.csv created
- [ ] Step 4 (Analysis): All 3 outputs created (2 PNG, 1 MD)

### 12.2 Quality Metrics

- [ ] Classification accuracy >80%
- [ ] All plots meet resolution requirements
- [ ] analysis.md contains all required sections
- [ ] Code files ≤250 lines each
- [ ] All functions have type hints and docstrings
- [ ] Execution time <10 seconds

### 12.3 Usability Metrics

- [ ] Single command execution works
- [ ] Clear console progress messages
- [ ] No manual intervention required
- [ ] README provides clear instructions

---

## 13. Open Questions & Decisions

### 13.1 Resolved Decisions

✓ **Q**: Use custom gradient descent or sklearn?
**A**: Custom implementation for iteration tracking

✓ **Q**: Monolithic or modular structure?
**A**: Modular (4-5 files) for maintainability

✓ **Q**: Where to save output files?
**A**: Root directory (as per PRD file structure)

### 13.2 Implementation Flexibility

**Configurable Choices:**
- Exact gradient descent algorithm (batch vs. stochastic vs. mini-batch)
- Color scheme for plots
- Specific markdown formatting style
- Table formatting in analysis.md
- Logging verbosity level

**Non-Negotiable (Per PRD):**
- Must use sigmoid activation
- Must use logistic regression
- Must track iteration metrics
- Must generate all 5 outputs
- Files must be ≤250 lines

---

## 14. Next Steps

1. **Review this planning document** with stakeholders/team
2. **Set up development environment** (Phase 1)
3. **Begin implementation** following the phase plan
4. **Track progress** using TODO.md checklist
5. **Test frequently** after each phase
6. **Document as you go** to maintain clarity

---

## Document Control

**Version**: 1.0
**Created**: 2025-11-15
**Last Updated**: 2025-11-15
**Status**: Ready for Implementation
**Related Documents**: PRD.md, TODO.md, CLAUDE.md
