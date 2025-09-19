# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Random Forest Scaling Laws Research Framework that analyzes how Random Forest performance scales with computational resources. The project systematically analyzes training time, memory usage, and prediction performance across various dataset sizes and Random Forest hyperparameters.

## Key Commands

### Running Experiments
```bash
# Run with default configuration
python main.py run

# Create custom configuration
python main.py create-config --output my_config.json
python main.py run --config my_config.json --output my_results

# Test framework functionality
python test_framework.py
```

### Development Workflow
```bash
# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook for interactive analysis
jupyter notebook scaling_laws_research/notebooks/random_forest_scaling_demo.ipynb
```

## Architecture

The codebase implements a comprehensive experiment framework following this architecture:

1. **Configuration System** (`scaling_laws_research/utils/config.py`): Uses dataclass-based configuration with `ExperimentConfig`, `RandomForestConfig`, and `DataConfig` classes. Supports JSON serialization for reproducibility.

2. **Experiment Runner** (`scaling_laws_research/experiments/scaling_experiments.py`): The `ScalingExperiment` class orchestrates experiments:
   - `run_data_scaling_experiment()`: Tests scaling with dataset size variations
   - `run_parameter_scaling_experiment()`: Tests scaling with hyperparameter variations
   - `run_full_experiment()`: Combines both experiments with result aggregation

3. **Analysis Pipeline** (`scaling_laws_research/analysis/scaling_laws.py`): The `ScalingLawAnalyzer` class extracts scaling laws using power-law fitting, providing insights on how performance metrics scale with various inputs.

4. **Visualization Suite** (`scaling_laws_research/visualizations/scaling_plots.py`): The `ScalingPlotter` class generates comprehensive plots including log-log scaling plots, parameter heatmaps, and performance comparisons.

5. **Data Generation** (`scaling_laws_research/utils/data_generator.py`): Handles synthetic dataset generation for controlled experiments with configurable complexity.

6. **Performance Tracking** (`scaling_laws_research/utils/performance_tracker.py`): Monitors resource usage (CPU, memory, time) during experiments using context managers.

## Key Experiment Parameters

The framework tests scaling across:
- **Data dimensions**: Sample counts (1K-16K) and feature counts (20-160)
- **Random Forest parameters**: n_estimators, max_depth, min_samples_split, min_samples_leaf
- **Metrics tracked**: training_time, prediction_time, memory_usage, accuracy/mse

## Output Structure

Experiments generate results in the specified output directory:
- `scaling_results.csv`: Raw experimental data with all metrics
- `experiment_config.json`: Configuration for reproducibility
- `scaling_analysis_report.txt`: Statistical analysis and scaling law extractions
- `*.png`: Visualization plots (data scaling, parameter scaling, performance comparisons)