# Random Forest Scaling Laws Research

A comprehensive framework for studying how Random Forest performance scales with different computational resources. This project systematically analyzes training time, memory usage, and prediction performance across various dataset sizes and Random Forest hyperparameters.

## Project Overview

This research framework helps answer critical questions about Random Forest scalability:

- How does training time scale with dataset size (samples and features)?
- What is the relationship between model complexity and computational resources?
- How do different Random Forest parameters affect scaling behavior?
- What are the practical limits for Random Forest deployment at scale?

## Project Structure

```
scaling_laws_research/
├── data/                   # Data generation and loading utilities
│   ├── __init__.py
│   ├── data_generator.py   # Synthetic dataset generation
│   └── data_loader.py      # Benchmark dataset loading
├── experiments/            # Experiment framework
│   ├── __init__.py
│   └── scaling_experiments.py  # Core experiment runner
├── analysis/               # Analysis and scaling law extraction
│   ├── __init__.py
│   └── scaling_laws.py     # Scaling law analyzer
├── visualizations/         # Plotting and visualization tools
│   ├── __init__.py
│   └── scaling_plots.py    # Comprehensive plotting suite
├── utils/                  # Utility classes and configuration
│   ├── __init__.py
│   ├── config.py          # Experiment configuration
│   ├── data_generator.py  # Data generation utilities
│   └── performance_tracker.py  # Resource monitoring
├── notebooks/             # Example notebooks and tutorials
│   └── random_forest_scaling_demo.ipynb
└── results/               # Output directory for results
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/elroy-galbraith/ml_scaling.git
cd ml_scaling
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Your First Experiment

1. **Using the CLI (Recommended)**:
```bash
# Run with default configuration
python main.py run

# Create a custom configuration
python main.py create-config --output my_config.json
# Edit my_config.json as needed, then run:
python main.py run --config my_config.json --output my_results
```

2. **Using Python directly**:
```python
from scaling_laws_research.utils.config import ExperimentConfig
from scaling_laws_research.experiments.scaling_experiments import ScalingExperiment

# Create configuration
config = ExperimentConfig(
    name="my_scaling_experiment",
    output_dir="results/my_experiment"
)

# Run experiment
experiment = ScalingExperiment(config)
results_df = experiment.run_full_experiment()
```

3. **Using the Jupyter Notebook**:
```bash
jupyter notebook scaling_laws_research/notebooks/random_forest_scaling_demo.ipynb
```

## Key Features

### 🔬 Comprehensive Scaling Analysis
- **Data Scaling**: Analyzes how performance scales with dataset size (samples and features)
- **Parameter Scaling**: Studies the impact of Random Forest hyperparameters
- **Resource Monitoring**: Tracks CPU usage, memory consumption, and execution time
- **Performance Metrics**: Measures accuracy, training time, and prediction time

### 📊 Advanced Visualizations
- Log-log scaling plots with power-law fits
- Parameter sensitivity heatmaps
- Performance comparison charts
- Resource utilization analysis

### 🧮 Mathematical Analysis
- Power-law fitting with statistical validation
- Scaling exponent extraction
- R-squared goodness-of-fit analysis
- Automated insight generation

### ⚙️ Flexible Configuration
- JSON-based experiment configuration
- Customizable parameter grids
- Support for both classification and regression
- Extensible design for new metrics

## Example Results

The framework automatically generates scaling laws like:

```
Training Time Analysis:
• Samples: O(n^1.05) - Linear scaling (efficient)
• Features: O(d^1.23) - Super-linear scaling (expensive)
• Estimators: O(trees^0.98) - Nearly linear scaling

Memory Usage Analysis:
• Samples: O(n^1.15) - Slightly super-linear
• Features: O(d^1.45) - Super-linear (memory-intensive)
```

## Configuration Options

### Data Configuration
```python
config.data.base_samples = 1000        # Base dataset size
config.data.base_features = 20         # Base feature count
config.data.sample_scales = [1,2,4,8]  # Sample scaling factors
config.data.feature_scales = [1,2,4]   # Feature scaling factors
config.data.task_type = "classification"  # or "regression"
```

### Random Forest Configuration
```python
config.random_forest.n_estimators = [10, 50, 100, 200]
config.random_forest.max_depth = [None, 5, 10, 20]
config.random_forest.min_samples_split = [2, 5, 10]
config.random_forest.min_samples_leaf = [1, 2, 4]
```

## Output Files

Each experiment generates:
- `scaling_results.csv` - Raw experimental data
- `experiment_config.json` - Experiment configuration
- `scaling_analysis_report.txt` - Detailed analysis report
- `*.png` - Visualization plots
- Optional: Model pickles and predictions

## Testing

Verify the framework works correctly:
```bash
python test_framework.py
```

## Advanced Usage

### Custom Metrics
Extend the framework with custom performance metrics by modifying the `ScalingExperiment` class.

### Real Datasets
Use benchmark datasets instead of synthetic data:
```python
from scaling_laws_research.data.data_loader import load_benchmark_datasets
datasets = load_benchmark_datasets()
```

### Parallel Execution
The framework supports parallel Random Forest training via scikit-learn's `n_jobs` parameter.

## Research Applications

This framework is useful for:
- **Resource Planning**: Predicting computational requirements for large-scale ML projects
- **Cost Analysis**: Understanding the relationship between model complexity and computational cost
- **Performance Optimization**: Finding optimal hyperparameter ranges for different scales
- **Academic Research**: Studying fundamental scaling properties of ensemble methods

## Contributing

Contributions are welcome! Key areas for improvement:
- Additional ML algorithms beyond Random Forest
- More sophisticated scaling law models
- Integration with cloud computing platforms
- Real-time monitoring capabilities

## Citation

If you use this framework in your research, please cite:
```
@software{ml_scaling_framework,
  title={Random Forest Scaling Laws Research Framework},
  author={ML Scaling Research Team},
  year={2024},
  url={https://github.com/elroy-galbraith/ml_scaling}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
