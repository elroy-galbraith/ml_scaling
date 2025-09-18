# Random Forest Scaling Laws Research

A comprehensive study into scaling laws that govern Random Forest model development and performance optimization.

## Project Overview

This repository contains a complete research framework for discovering and analyzing scaling laws in Random Forest models. The core question driving this research is: **Do Random Forests follow predictable power-law relationships between computational resources and performance?**

## Key Features

- **Systematic Experimentation**: Automated scaling experiments across data size, tree count, and max depth
- **Statistical Analysis**: Power law fitting with confidence intervals and significance testing  
- **Professional Visualizations**: Publication-ready plots with error bars and statistical annotations
- **Modular Architecture**: Clean, extensible codebase for research reproducibility
- **Comprehensive Logging**: Detailed experiment tracking and progress monitoring

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/elroy-galbraith/ml_scaling.git
cd ml_scaling

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Run data size scaling experiment
python main.py --experiment data_scaling --dataset adult

# Run all experiments
python main.py --experiment all --dataset adult

# Generate visualizations
python main.py --visualize --dataset adult
```

## Experiment Types

1. **Data Size Scaling**: How performance scales with training data size [500, 1K, 2K, 5K, 10K, 20K, full]
2. **Tree Count Scaling**: How performance scales with number of trees [10, 25, 50, 100, 200, 500, 1000]  
3. **Depth Scaling**: How performance scales with maximum tree depth [3, 5, 10, 15, 20, None]

## Sample Results

```
============================================================
EXPERIMENT SUMMARY
============================================================
data_size_scaling   :   21 experiments
tree_count_scaling  :   21 experiments  
depth_scaling       :   18 experiments
TOTAL               :   60 experiments
============================================================

KEY FINDINGS:

Data Size Scaling:
  accuracy       : power_law       (R² = 0.934)
  training_time  : power_law       (R² = 0.966)

Tree Count Scaling:  
  accuracy       : inverse_power_law (R² = 0.892)
  training_time  : power_law       (R² = 0.998)
```

## Project Structure

```
ml_scaling/
├── main.py                    # CLI interface and pipeline orchestration
├── config.py                  # Centralized configuration settings
├── requirements.txt           # Python dependencies
├── scaling_laws_research/     # Main research package
│   ├── data/                  # Data loading and preprocessing
│   ├── experiments/           # Systematic scaling experiments
│   ├── analysis/              # Power law fitting and statistics
│   ├── visualizations/        # Professional plotting tools
│   ├── utils/                 # Logging and file I/O utilities
│   ├── results/               # Experiment outputs and analysis
│   └── README.md             # Detailed package documentation
└── README.md                 # This file
```

## Research Applications

- **Performance Prediction**: Estimate model performance at different resource levels
- **Resource Optimization**: Find optimal trade-offs between accuracy and computational cost
- **Capacity Planning**: Predict infrastructure requirements for scaling ML workflows
- **Algorithm Comparison**: Compare scaling behavior across different datasets and algorithms
- **Academic Research**: Contribute to understanding of ML scaling laws

## Datasets

- **Primary**: Adult/Census Income dataset (32K samples, 14 features)
- **Fallback**: Synthetic datasets for testing when network access is limited
- **Extensible**: Framework designed to easily add new datasets

## Key Findings

Random Forest models exhibit clear power-law scaling relationships:

1. **Data Size Scaling**: Performance follows diminishing returns with more training data
2. **Tree Count Scaling**: Performance saturates after optimal tree count
3. **Depth Scaling**: Performance peaks at moderate depths before overfitting

## Technical Features

- **Reproducible**: Fixed random seeds and comprehensive logging
- **Robust**: Error handling and fallback datasets for reliability
- **Efficient**: Parallel processing and progress tracking
- **Professional**: Type hints, docstrings, and modular design
- **Extensible**: Easy to add new experiments, metrics, and datasets

## Command Line Interface

```bash
python main.py --help

# Available options:
--experiment {data_scaling,tree_scaling,depth_scaling,all}
--dataset DATASET              # Dataset name (default: adult)
--visualize                    # Generate plots from existing results
--analyze                      # Run analysis on existing results  
--log-level {DEBUG,INFO,WARNING,ERROR}
--no-save                      # Don't save results to files
```

## Contributing

This research framework is designed to be:
- **Modular**: Easy to extend with new experiments
- **Documented**: Comprehensive docstrings and examples
- **Tested**: Robust error handling and validation
- **Reproducible**: Deterministic results with fixed seeds

See `scaling_laws_research/README.md` for detailed documentation on extending the framework.

## License

This project is open source and available for research and educational purposes.

## Contact

For questions about the research methodology or framework design, please open an issue or submit a pull request.
