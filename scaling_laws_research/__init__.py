"""
Random Forest Scaling Laws Research Project

A comprehensive framework for studying how Random Forest performance 
scales with different computational resources.
"""

__version__ = "0.1.0"
__author__ = "ML Scaling Research Team"

from . import utils
from . import experiments
from . import analysis
from . import visualizations

__all__ = ["utils", "experiments", "analysis", "visualizations"]