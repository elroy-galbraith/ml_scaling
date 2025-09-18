"""
Random Forest Scaling Laws Research Package.

A comprehensive framework for studying scaling laws in Random Forest models,
exploring how performance scales with computational resources.
"""

__version__ = "0.1.0"
__author__ = "ML Scaling Research Team"

from .data import DataLoader, load_and_prepare_data
from .experiments import ScalingExperiments
from .analysis import PowerLawFitter, analyze_scaling_results
from .visualizations import ScalingPlotter, create_all_visualizations
from .utils import setup_logging, get_logger, ResultsIO

__all__ = [
    "DataLoader",
    "load_and_prepare_data", 
    "ScalingExperiments",
    "PowerLawFitter",
    "analyze_scaling_results",
    "ScalingPlotter", 
    "create_all_visualizations",
    "setup_logging",
    "get_logger",
    "ResultsIO"
]