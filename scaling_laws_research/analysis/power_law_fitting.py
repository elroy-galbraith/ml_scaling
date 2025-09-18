"""
Power law fitting and statistical analysis for scaling experiments.

This module provides functions to fit power laws to scaling data, calculate
confidence intervals, and perform statistical significance testing.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.optimize import curve_fit
from scipy import stats
import warnings

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config

# Set up logging
logger = logging.getLogger(__name__)

# Suppress warnings from curve fitting
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')


def power_law_function(x: np.ndarray, a: float, b: float, c: float = 0) -> np.ndarray:
    """
    Power law function: f(x) = a * x^(-b) + c
    
    Args:
        x: Input values (resource parameter)
        a: Amplitude parameter
        b: Exponent parameter (positive for decreasing, negative for increasing)
        c: Offset parameter
        
    Returns:
        Power law values
    """
    return a * np.power(x, -b) + c


def inverse_power_law_function(x: np.ndarray, a: float, b: float, c: float = 0) -> np.ndarray:
    """
    Inverse power law function: f(x) = a * x^b + c
    
    Args:
        x: Input values (resource parameter)
        a: Amplitude parameter
        b: Exponent parameter
        c: Offset parameter
        
    Returns:
        Inverse power law values
    """
    return a * np.power(x, b) + c


class PowerLawFitter:
    """Class for fitting power laws to scaling data."""
    
    def __init__(self):
        """Initialize the power law fitter."""
        self.fit_results = {}
    
    def prepare_data(
        self, 
        results: List[Dict[str, Any]], 
        x_param: str, 
        y_metric: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for power law fitting.
        
        Args:
            results: List of experiment results
            x_param: Parameter name for x-axis (e.g., 'data_size', 'n_estimators')
            y_metric: Metric name for y-axis (e.g., 'accuracy', 'f1')
            
        Returns:
            Tuple of (x_values, y_means, y_stds)
        """
        df = pd.DataFrame(results)
        
        # Group by x parameter and calculate statistics
        grouped = df.groupby(x_param)[y_metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Handle None values (e.g., max_depth=None)
        if x_param == 'max_depth':
            # Replace None with a large number for fitting purposes
            grouped[x_param] = grouped[x_param].fillna(1000)
        
        x_values = grouped[x_param].values.astype(float)
        y_means = grouped['mean'].values
        y_stds = grouped['std'].fillna(0).values  # Fill NaN std with 0
        
        # Filter out invalid values
        valid_mask = (x_values > 0) & (y_means > 0) & np.isfinite(x_values) & np.isfinite(y_means)
        
        return x_values[valid_mask], y_means[valid_mask], y_stds[valid_mask]
    
    def fit_power_law(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        y_std: Optional[np.ndarray] = None,
        use_inverse: bool = False
    ) -> Dict[str, Any]:
        """
        Fit a power law to the data.
        
        Args:
            x: Independent variable values
            y: Dependent variable values
            y_std: Standard deviations of y values (for weighting)
            use_inverse: Whether to use inverse power law (x^b instead of x^(-b))
            
        Returns:
            Dictionary containing fit results
        """
        if len(x) < 3:
            logger.warning("Not enough data points for power law fitting")
            return {"success": False, "error": "Insufficient data points"}
        
        # Choose function
        func = inverse_power_law_function if use_inverse else power_law_function
        func_name = "inverse_power_law" if use_inverse else "power_law"
        
        # Set bounds
        bounds = config.POWER_LAW_BOUNDS
        lower_bounds = [bounds["a"][0], bounds["b"][0], bounds["c"][0]]
        upper_bounds = [bounds["a"][1], bounds["b"][1], bounds["c"][1]]
        
        # Initial guess
        a_init = np.mean(y)
        b_init = 0.1 if use_inverse else -0.1
        c_init = 0.0
        p0 = [a_init, b_init, c_init]
        
        try:
            # Fit with error weighting if available
            sigma = 1.0 / (y_std + 1e-10) if y_std is not None else None
            
            popt, pcov = curve_fit(
                func, x, y, 
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                sigma=sigma,
                absolute_sigma=False,
                maxfev=10000
            )
            
            # Calculate fit quality metrics
            y_pred = func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate parameter uncertainties
            param_std = np.sqrt(np.diag(pcov))
            
            # Calculate AIC and BIC
            n = len(x)
            mse = ss_res / n
            aic = n * np.log(mse) + 2 * 3  # 3 parameters
            bic = n * np.log(mse) + np.log(n) * 3
            
            fit_result = {
                "success": True,
                "function": func_name,
                "parameters": {
                    "a": float(popt[0]),
                    "b": float(popt[1]),
                    "c": float(popt[2])
                },
                "parameter_std": {
                    "a_std": float(param_std[0]),
                    "b_std": float(param_std[1]),
                    "c_std": float(param_std[2])
                },
                "r_squared": float(r_squared),
                "aic": float(aic),
                "bic": float(bic),
                "mse": float(mse),
                "n_points": int(n)
            }
            
            logger.debug(f"Power law fit successful: R² = {r_squared:.4f}")
            return fit_result
            
        except Exception as e:
            logger.warning(f"Power law fitting failed: {e}")
            return {"success": False, "error": str(e)}
    
    def fit_scaling_relationship(
        self,
        results: List[Dict[str, Any]],
        x_param: str,
        y_metric: str,
        try_both_forms: bool = True
    ) -> Dict[str, Any]:
        """
        Fit scaling relationship and determine best form.
        
        Args:
            results: List of experiment results
            x_param: Parameter name for x-axis
            y_metric: Metric name for y-axis
            try_both_forms: Whether to try both power law forms
            
        Returns:
            Dictionary containing best fit results
        """
        logger.info(f"Fitting scaling relationship: {y_metric} vs {x_param}")
        
        # Prepare data
        x, y, y_std = self.prepare_data(results, x_param, y_metric)
        
        if len(x) < 3:
            return {"success": False, "error": "Insufficient data points"}
        
        fits = {}
        
        # Try regular power law
        fit_regular = self.fit_power_law(x, y, y_std, use_inverse=False)
        if fit_regular["success"]:
            fits["power_law"] = fit_regular
        
        # Try inverse power law if requested
        if try_both_forms:
            fit_inverse = self.fit_power_law(x, y, y_std, use_inverse=True)
            if fit_inverse["success"]:
                fits["inverse_power_law"] = fit_inverse
        
        if not fits:
            return {"success": False, "error": "All fitting attempts failed"}
        
        # Select best fit based on R-squared
        best_fit_name = max(fits.keys(), key=lambda k: fits[k]["r_squared"])
        best_fit = fits[best_fit_name]
        
        # Add comparison information
        result = {
            "success": True,
            "x_param": x_param,
            "y_metric": y_metric,
            "best_fit": best_fit_name,
            "all_fits": fits,
            "data": {
                "x": x.tolist(),
                "y": y.tolist(),
                "y_std": y_std.tolist()
            },
            **best_fit  # Include best fit parameters at top level
        }
        
        logger.info(f"Best fit: {best_fit_name} (R² = {best_fit['r_squared']:.4f})")
        return result
    
    def generate_predictions(
        self, 
        fit_result: Dict[str, Any], 
        x_range: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions from a fitted power law.
        
        Args:
            fit_result: Fit result dictionary
            x_range: X values for prediction (if None, use original data range)
            
        Returns:
            Tuple of (x_pred, y_pred)
        """
        if not fit_result["success"]:
            raise ValueError("Cannot generate predictions from failed fit")
        
        if x_range is None:
            x_data = np.array(fit_result["data"]["x"])
            x_min, x_max = x_data.min(), x_data.max()
            x_range = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        
        # Get parameters
        params = fit_result["parameters"]
        func_name = fit_result["function"]
        
        # Generate predictions
        if func_name == "power_law":
            y_pred = power_law_function(x_range, params["a"], params["b"], params["c"])
        elif func_name == "inverse_power_law":
            y_pred = inverse_power_law_function(x_range, params["a"], params["b"], params["c"])
        else:
            raise ValueError(f"Unknown function: {func_name}")
        
        return x_range, y_pred
    
    def calculate_confidence_intervals(
        self,
        fit_result: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for fitted parameters.
        
        Args:
            fit_result: Fit result dictionary
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary of parameter confidence intervals
        """
        if not fit_result["success"]:
            return {}
        
        # Calculate t-value for confidence interval
        n = fit_result["n_points"]
        dof = n - 3  # 3 parameters
        alpha = 1 - confidence_level
        t_val = stats.t.ppf(1 - alpha/2, dof)
        
        confidence_intervals = {}
        for param in ["a", "b", "c"]:
            param_val = fit_result["parameters"][param]
            param_std = fit_result["parameter_std"][f"{param}_std"]
            margin = t_val * param_std
            
            confidence_intervals[param] = (
                param_val - margin,
                param_val + margin
            )
        
        return confidence_intervals
    
    def statistical_significance_test(
        self, 
        fit_result: Dict[str, Any],
        test_parameter: str = "b"
    ) -> Dict[str, float]:
        """
        Test statistical significance of fitted parameters.
        
        Args:
            fit_result: Fit result dictionary
            test_parameter: Parameter to test (default: 'b' for exponent)
            
        Returns:
            Dictionary containing test statistics
        """
        if not fit_result["success"]:
            return {}
        
        # Get parameter value and standard error
        param_val = fit_result["parameters"][test_parameter]
        param_std = fit_result["parameter_std"][f"{test_parameter}_std"]
        
        # T-test against null hypothesis (parameter = 0)
        t_stat = param_val / param_std if param_std > 0 else 0
        dof = fit_result["n_points"] - 3  # 3 parameters
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))  # Two-tailed test
        
        return {
            "parameter": test_parameter,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01
        }


def analyze_scaling_results(
    results: List[Dict[str, Any]], 
    experiment_type: str
) -> Dict[str, Any]:
    """
    Comprehensive analysis of scaling experiment results.
    
    Args:
        results: List of experiment results
        experiment_type: Type of scaling experiment
        
    Returns:
        Dictionary containing comprehensive analysis
    """
    logger.info(f"Analyzing scaling results for {experiment_type}")
    
    fitter = PowerLawFitter()
    analysis_results = {
        "experiment_type": experiment_type,
        "n_experiments": len(results),
        "scaling_relationships": {}
    }
    
    # Determine x parameter based on experiment type
    if experiment_type == "data_size_scaling":
        x_param = "data_size"
    elif experiment_type == "tree_count_scaling":
        x_param = "n_estimators"
    elif experiment_type == "depth_scaling":
        x_param = "max_depth"
    else:
        logger.error(f"Unknown experiment type: {experiment_type}")
        return analysis_results
    
    # Analyze each metric
    for metric in config.METRICS:
        if metric in ["training_time", "prediction_time"]:
            # For time metrics, we expect increasing relationship
            fit_result = fitter.fit_scaling_relationship(
                results, x_param, metric, try_both_forms=True
            )
        else:
            # For performance metrics, we expect decreasing relationship with more data
            # but increasing with more trees/depth
            fit_result = fitter.fit_scaling_relationship(
                results, x_param, metric, try_both_forms=True
            )
        
        if fit_result["success"]:
            # Add confidence intervals and significance tests
            fit_result["confidence_intervals"] = fitter.calculate_confidence_intervals(fit_result)
            fit_result["significance_test"] = fitter.statistical_significance_test(fit_result)
            
            analysis_results["scaling_relationships"][metric] = fit_result
            
            logger.info(f"  {metric}: {fit_result['best_fit']} "
                       f"(R² = {fit_result['r_squared']:.4f})")
        else:
            logger.warning(f"  {metric}: Fitting failed")
            analysis_results["scaling_relationships"][metric] = fit_result
    
    return analysis_results