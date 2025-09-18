# Random Forest Scaling Laws Research

This package provides a comprehensive framework for studying scaling laws in Random Forest models, exploring how performance scales with computational resources.

## Core Research Question

Do Random Forests follow predictable power-law relationships between resources (training data size, number of trees, max depth) and performance?

## Package Structure

```
scaling_laws_research/
├── data/              # Data loading and preprocessing
├── experiments/       # Systematic scaling experiments
├── analysis/          # Power law fitting and statistical analysis
├── visualizations/    # Professional plotting tools
├── results/           # Experiment results and outputs
├── notebooks/         # Jupyter notebooks for analysis
└── utils/            # Utility functions and helpers
```

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Experiments

1. **Run data size scaling experiment:**
```bash
python main.py --experiment data_scaling --dataset adult
```

2. **Run all experiments:**
```bash
python main.py --experiment all --dataset adult
```

3. **Generate visualizations from existing results:**
```bash
python main.py --visualize --dataset adult
```

4. **Run analysis only:**
```bash
python main.py --analyze --dataset adult
```

### Command Line Options

- `--experiment`: Type of experiment (`data_scaling`, `tree_scaling`, `depth_scaling`, `all`)
- `--dataset`: Dataset to use (default: `adult`)
- `--visualize`: Generate visualizations from existing results
- `--analyze`: Run analysis on existing results
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--no-save`: Don't save results to files

## Experiments

### 1. Data Size Scaling
- **Purpose**: Test how performance scales with training data size
- **Variables**: Training data sizes: [500, 1K, 2K, 5K, 10K, 20K, full]
- **Fixed Parameters**: n_estimators=100, max_depth=None

### 2. Tree Count Scaling  
- **Purpose**: Test how performance scales with number of trees
- **Variables**: Tree counts: [10, 25, 50, 100, 200, 500, 1000]
- **Fixed Parameters**: Full dataset, max_depth=None

### 3. Depth Scaling
- **Purpose**: Test how performance scales with maximum tree depth
- **Variables**: Max depths: [3, 5, 10, 15, 20, None]
- **Fixed Parameters**: Full dataset, n_estimators=100

## Metrics Tracked

For each experiment, the following metrics are measured:
- **Performance**: accuracy, precision, recall, F1-score
- **Computational**: training_time, prediction_time

## Analysis

### Power Law Fitting

The framework fits power law relationships of the form:
- `performance = a * resource^(-b) + c` (decreasing performance)
- `performance = a * resource^b + c` (increasing performance)

Where:
- `a` = amplitude parameter
- `b` = scaling exponent  
- `c` = offset parameter

### Statistical Analysis

- R-squared values for fit quality
- Confidence intervals for parameters
- Statistical significance testing
- AIC/BIC for model comparison

## Results Structure

Results are automatically saved in structured format:

```
results/
├── {dataset}/
│   ├── data_size_scaling/
│   │   ├── data_size_scaling_results.csv
│   │   ├── data_size_scaling_results.json
│   │   ├── data_size_scaling_analysis.json
│   │   └── data_size_scaling_metadata.json
│   ├── tree_count_scaling/
│   ├── depth_scaling/
│   └── summary/
│       └── complete_summary.json
└── logs/
    └── scaling_research_{dataset}.log
```

## Visualizations

Professional plots are generated including:
- Individual scaling relationships with power law fits
- Summary plots across all experiments
- Power law parameter comparisons
- Error bars and confidence intervals

Plots are saved in both PNG and SVG formats in:
```
visualizations/plots/{dataset}/
```

## Example Output

After running a data scaling experiment, you'll see output like:

```
============================================================
EXPERIMENT SUMMARY  
============================================================
data_size_scaling   :   21 experiments
TOTAL               :   21 experiments
============================================================

KEY FINDINGS:

Data Size Scaling:
  accuracy       : power_law       (R² = 0.934)
  precision      : power_law       (R² = 0.934)  
  recall         : power_law       (R² = 0.934)
  f1             : power_law       (R² = 0.934)
  training_time  : power_law       (R² = 0.966)
  prediction_time: power_law       (R² = -0.025)
```

## Programmatic Usage

```python
from scaling_laws_research import ScalingExperiments, analyze_scaling_results
from scaling_laws_research.visualizations import create_all_visualizations

# Run experiments
experiments = ScalingExperiments("adult")
data_results = experiments.data_size_scaling()
tree_results = experiments.tree_count_scaling()
depth_results = experiments.depth_scaling()

# Analyze results  
all_results = {
    "data_size_scaling": data_results,
    "tree_count_scaling": tree_results, 
    "depth_scaling": depth_results
}

analysis = {}
for exp_type, results in all_results.items():
    analysis[exp_type] = analyze_scaling_results(results, exp_type)

# Generate visualizations
figures = create_all_visualizations(all_results, analysis, "adult")
```

## Configuration

Key settings can be modified in `config.py`:
- Random seeds for reproducibility
- Data sizes and parameter ranges
- Number of experiment repetitions
- Visualization styling
- File paths

## Research Applications

This framework is designed for:
- Understanding Random Forest scaling behavior
- Optimizing resource allocation for ML workflows
- Comparing scaling laws across different datasets
- Academic research on ML scaling laws
- Performance prediction and capacity planning

## Extensibility

The modular design allows easy extension for:
- Additional datasets
- New experiment types
- Different ML algorithms
- Custom metrics
- Alternative analysis methods

## Dependencies

- scikit-learn: Random Forest implementation
- pandas: Data manipulation
- numpy: Numerical operations
- matplotlib/seaborn: Visualizations
- scipy: Statistical analysis
- tqdm: Progress tracking