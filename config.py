"""
Configuration module for Random Forest scaling laws research.

This module contains all hyperparameters, experimental settings, and paths
used throughout the research pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List, Any


class Config:
    """Central configuration class for the scaling laws research."""
    
    # Random seeds for reproducibility
    RANDOM_SEEDS = [42, 123, 456]
    
    # Data configuration
    DATASET_NAME = "adult"  # Default to Adult/Census dataset
    TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
    STRATIFY = True
    
    # Data size scaling parameters
    DATA_SIZES = [500, 1000, 2000, 5000, 10000, 20000, "full"]
    
    # Tree count scaling parameters
    TREE_COUNTS = [10, 25, 50, 100, 200, 500, 1000]
    
    # Max depth scaling parameters
    MAX_DEPTHS = [3, 5, 10, 15, 20, None]
    
    # Random Forest default parameters
    RF_DEFAULT_PARAMS = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
        "oob_score": True
    }
    
    # Performance metrics to track
    METRICS = ["accuracy", "precision", "recall", "f1", "training_time", "prediction_time"]
    
    # Experiment repetitions
    N_REPETITIONS = 3
    
    # Power law fitting parameters
    POWER_LAW_BOUNDS = {
        "a": (0.001, 10.0),      # Amplitude
        "b": (-2.0, 2.0),        # Exponent
        "c": (0.0, 1.0)          # Offset
    }
    
    # Visualization parameters
    PLOT_CONFIG = {
        "figure_size": (10, 8),
        "dpi": 300,
        "style": "seaborn-v0_8-whitegrid",
        "font_size": 12,
        "title_size": 14,
        "label_size": 12,
        "legend_size": 10,
        "colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        "save_formats": ["png", "svg"]
    }
    
    # File paths
    BASE_DIR = Path(__file__).parent.parent.absolute()
    DATA_DIR = BASE_DIR / "scaling_laws_research" / "data"
    RESULTS_DIR = BASE_DIR / "scaling_laws_research" / "results"
    PLOTS_DIR = BASE_DIR / "scaling_laws_research" / "visualizations" / "plots"
    NOTEBOOKS_DIR = BASE_DIR / "scaling_laws_research" / "notebooks"
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [cls.DATA_DIR, cls.RESULTS_DIR, cls.PLOTS_DIR, cls.NOTEBOOKS_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_results_path(cls, dataset: str, experiment_type: str) -> Path:
        """Get the results path for a specific dataset and experiment type."""
        results_path = cls.RESULTS_DIR / dataset / experiment_type
        results_path.mkdir(parents=True, exist_ok=True)
        return results_path
    
    @classmethod
    def get_plots_path(cls, dataset: str) -> Path:
        """Get the plots path for a specific dataset."""
        plots_path = cls.PLOTS_DIR / dataset
        plots_path.mkdir(parents=True, exist_ok=True)
        return plots_path


# Export the config instance
config = Config()