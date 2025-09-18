"""Analysis package for power law fitting and statistical analysis."""

from .power_law_fitting import (
    PowerLawFitter, 
    power_law_function, 
    inverse_power_law_function,
    analyze_scaling_results
)

__all__ = [
    "PowerLawFitter", 
    "power_law_function", 
    "inverse_power_law_function",
    "analyze_scaling_results"
]