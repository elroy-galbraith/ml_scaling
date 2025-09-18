"""
Visualization tools for Random Forest scaling laws research.

This module provides functions to create professional plots for scaling laws,
including error bars, log-scale axes, and publication-ready styling.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config
from scaling_laws_research.analysis import PowerLawFitter, power_law_function, inverse_power_law_function

# Set up logging
logger = logging.getLogger(__name__)

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set up matplotlib and seaborn style
plt.style.use('default')  # Use default instead of seaborn style
sns.set_palette("husl")


class ScalingPlotter:
    """Class for creating scaling law visualizations."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize the scaling plotter.
        
        Args:
            save_dir: Directory to save plots (if None, uses config default)
        """
        self.save_dir = save_dir
        self.plot_config = config.PLOT_CONFIG
        
        # Configure matplotlib
        plt.rcParams.update({
            'font.size': self.plot_config['font_size'],
            'axes.titlesize': self.plot_config['title_size'],
            'axes.labelsize': self.plot_config['label_size'],
            'legend.fontsize': self.plot_config['legend_size'],
            'figure.dpi': self.plot_config['dpi']
        })
    
    def prepare_plot_data(
        self, 
        results: List[Dict[str, Any]], 
        x_param: str, 
        y_metric: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for plotting.
        
        Args:
            results: List of experiment results
            x_param: Parameter name for x-axis
            y_metric: Metric name for y-axis
            
        Returns:
            Tuple of (x_values, y_means, y_stds)
        """
        df = pd.DataFrame(results)
        
        # Group by x parameter and calculate statistics
        grouped = df.groupby(x_param)[y_metric].agg(['mean', 'std', 'count']).reset_index()
        
        # Handle None values (e.g., max_depth=None)
        if x_param == 'max_depth':
            # Replace None with a label for plotting
            grouped[x_param] = grouped[x_param].fillna("None")
            # For None values, use a special x position
            none_mask = grouped[x_param] == "None"
            if none_mask.any():
                max_val = grouped[grouped[x_param] != "None"][x_param].max()
                grouped.loc[none_mask, x_param] = max_val * 1.5
        
        x_values = grouped[x_param].values
        y_means = grouped['mean'].values
        y_stds = grouped['std'].fillna(0).values
        
        return x_values, y_means, y_stds
    
    def plot_scaling_relationship(
        self,
        results: List[Dict[str, Any]],
        x_param: str,
        y_metric: str,
        fit_result: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        save_name: Optional[str] = None,
        log_scale: bool = True
    ) -> plt.Figure:
        """
        Plot scaling relationship with optional power law fit.
        
        Args:
            results: List of experiment results
            x_param: Parameter name for x-axis
            y_metric: Metric name for y-axis
            fit_result: Power law fit result (optional)
            title: Plot title (if None, auto-generated)
            save_name: Name for saving plot (if None, auto-generated)
            log_scale: Whether to use log scale
            
        Returns:
            matplotlib Figure object
        """
        # Prepare data
        x, y_mean, y_std = self.prepare_plot_data(results, x_param, y_metric)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.plot_config['figure_size'])
        
        # Plot data points with error bars
        ax.errorbar(
            x, y_mean, yerr=y_std,
            fmt='o', capsize=5, capthick=2, markersize=8,
            color=self.plot_config['colors'][0],
            label='Experimental data',
            alpha=0.8
        )
        
        # Plot fit line if provided
        if fit_result and fit_result["success"]:
            fitter = PowerLawFitter()
            x_pred, y_pred = fitter.generate_predictions(fit_result)
            
            # Create label with fit parameters
            params = fit_result["parameters"]
            r_sq = fit_result["r_squared"]
            
            if fit_result["function"] == "power_law":
                label = f'Power law: y = {params["a"]:.3f} × x^{-params["b"]:.3f} + {params["c"]:.3f}\nR² = {r_sq:.3f}'
            else:
                label = f'Power law: y = {params["a"]:.3f} × x^{params["b"]:.3f} + {params["c"]:.3f}\nR² = {r_sq:.3f}'
            
            ax.plot(
                x_pred, y_pred,
                '-', linewidth=3,
                color=self.plot_config['colors'][1],
                label=label,
                alpha=0.9
            )
        
        # Set scales
        if log_scale:
            # Only use log scale if all values are positive
            if np.all(x > 0) and np.all(y_mean > 0):
                ax.set_xscale('log')
                ax.set_yscale('log')
            else:
                logger.warning("Cannot use log scale with non-positive values")
        
        # Labels and title
        ax.set_xlabel(self._format_parameter_name(x_param))
        ax.set_ylabel(self._format_metric_name(y_metric))
        
        if title is None:
            title = f'{self._format_metric_name(y_metric)} vs {self._format_parameter_name(x_param)}'
        ax.set_title(title, fontsize=self.plot_config['title_size'], fontweight='bold')
        
        # Grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if requested
        if save_name or self.save_dir:
            self._save_plot(fig, save_name or f"{x_param}_{y_metric}_scaling")
        
        return fig
    
    def plot_all_metrics_scaling(
        self,
        results: List[Dict[str, Any]],
        x_param: str,
        analysis_results: Optional[Dict[str, Any]] = None,
        title_prefix: str = "",
        save_prefix: str = ""
    ) -> Dict[str, plt.Figure]:
        """
        Plot scaling relationships for all metrics.
        
        Args:
            results: List of experiment results
            x_param: Parameter name for x-axis
            analysis_results: Analysis results with fits (optional)
            title_prefix: Prefix for plot titles
            save_prefix: Prefix for save names
            
        Returns:
            Dictionary of metric names to Figure objects
        """
        figures = {}
        
        for metric in config.METRICS:
            # Check if we have fit results for this metric
            fit_result = None
            if (analysis_results and 
                "scaling_relationships" in analysis_results and 
                metric in analysis_results["scaling_relationships"]):
                fit_result = analysis_results["scaling_relationships"][metric]
            
            # Create title and save name
            title = f"{title_prefix}{self._format_metric_name(metric)} vs {self._format_parameter_name(x_param)}"
            save_name = f"{save_prefix}{x_param}_{metric}_scaling"
            
            # Create plot
            try:
                fig = self.plot_scaling_relationship(
                    results, x_param, metric, fit_result, title, save_name
                )
                figures[metric] = fig
                logger.debug(f"Created plot for {metric}")
            except Exception as e:
                logger.warning(f"Failed to create plot for {metric}: {e}")
        
        return figures
    
    def plot_experiment_summary(
        self,
        all_results: Dict[str, List[Dict[str, Any]]],
        analysis_results: Dict[str, Dict[str, Any]],
        dataset_name: str = "adult"
    ) -> plt.Figure:
        """
        Create a summary plot with all scaling relationships.
        
        Args:
            all_results: Dictionary of experiment results
            analysis_results: Dictionary of analysis results
            dataset_name: Name of the dataset
            
        Returns:
            matplotlib Figure object
        """
        # Create subplots: 3 experiments × 2 metrics (accuracy and training_time)
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Random Forest Scaling Laws - {dataset_name.title()} Dataset', 
                    fontsize=16, fontweight='bold')
        
        experiments = [
            ("data_size_scaling", "data_size", "Data Size"),
            ("tree_count_scaling", "n_estimators", "Number of Trees"), 
            ("depth_scaling", "max_depth", "Max Depth")
        ]
        
        metrics = ["accuracy", "training_time"]
        metric_titles = ["Accuracy", "Training Time (s)"]
        
        for i, (exp_type, x_param, exp_title) in enumerate(experiments):
            if exp_type not in all_results:
                continue
                
            results = all_results[exp_type]
            
            for j, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
                ax = axes[i, j]
                
                # Prepare data
                x, y_mean, y_std = self.prepare_plot_data(results, x_param, metric)
                
                # Plot data
                ax.errorbar(
                    x, y_mean, yerr=y_std,
                    fmt='o', capsize=3, markersize=6,
                    color=self.plot_config['colors'][i],
                    alpha=0.8
                )
                
                # Plot fit if available
                if (exp_type in analysis_results and 
                    "scaling_relationships" in analysis_results[exp_type] and
                    metric in analysis_results[exp_type]["scaling_relationships"]):
                    
                    fit_result = analysis_results[exp_type]["scaling_relationships"][metric]
                    if fit_result["success"]:
                        fitter = PowerLawFitter()
                        x_pred, y_pred = fitter.generate_predictions(fit_result)
                        ax.plot(x_pred, y_pred, '-', linewidth=2,
                               color=self.plot_config['colors'][i], alpha=0.7)
                
                # Formatting
                ax.set_xlabel(exp_title)
                ax.set_ylabel(metric_title)
                ax.grid(True, alpha=0.3)
                
                # Use log scale for appropriate plots
                if metric in ["training_time"] or x_param in ["data_size", "n_estimators"]:
                    if np.all(x > 0) and np.all(y_mean > 0):
                        ax.set_xscale('log')
                        if metric == "training_time":
                            ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save summary plot
        if self.save_dir:
            self._save_plot(fig, f"{dataset_name}_scaling_summary")
        
        return fig
    
    def plot_power_law_parameters(
        self,
        analysis_results: Dict[str, Dict[str, Any]],
        dataset_name: str = "adult"
    ) -> plt.Figure:
        """
        Plot power law parameters across different experiments.
        
        Args:
            analysis_results: Dictionary of analysis results
            dataset_name: Name of the dataset
            
        Returns:
            matplotlib Figure object
        """
        # Extract power law exponents
        exponents = {}
        confidence_intervals = {}
        
        for exp_type, exp_results in analysis_results.items():
            if "scaling_relationships" not in exp_results:
                continue
            
            exponents[exp_type] = {}
            confidence_intervals[exp_type] = {}
            
            for metric, fit_result in exp_results["scaling_relationships"].items():
                if fit_result["success"]:
                    b_param = fit_result["parameters"]["b"]
                    exponents[exp_type][metric] = b_param
                    
                    if "confidence_intervals" in fit_result:
                        ci = fit_result["confidence_intervals"]["b"]
                        confidence_intervals[exp_type][metric] = ci
        
        if not exponents:
            logger.warning("No successful fits found for parameter plotting")
            return plt.figure()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for plotting
        exp_types = list(exponents.keys())
        metrics = config.METRICS
        
        x_pos = np.arange(len(exp_types))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            values = []
            errors = []
            
            for exp_type in exp_types:
                if metric in exponents[exp_type]:
                    values.append(exponents[exp_type][metric])
                    
                    # Calculate error from confidence interval
                    if (exp_type in confidence_intervals and 
                        metric in confidence_intervals[exp_type]):
                        ci = confidence_intervals[exp_type][metric]
                        error = (ci[1] - ci[0]) / 2
                        errors.append(error)
                    else:
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)
            
            # Plot bars
            ax.bar(
                x_pos + i * width, values,
                width, yerr=errors,
                label=self._format_metric_name(metric),
                alpha=0.8,
                capsize=3
            )
        
        # Formatting
        ax.set_xlabel('Experiment Type')
        ax.set_ylabel('Power Law Exponent (b)')
        ax.set_title(f'Power Law Exponents - {dataset_name.title()} Dataset')
        ax.set_xticks(x_pos + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels([self._format_experiment_name(exp) for exp in exp_types])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        if self.save_dir:
            self._save_plot(fig, f"{dataset_name}_power_law_parameters")
        
        return fig
    
    def _format_parameter_name(self, param: str) -> str:
        """Format parameter name for display."""
        format_map = {
            "data_size": "Training Data Size",
            "n_estimators": "Number of Trees",
            "max_depth": "Maximum Depth"
        }
        return format_map.get(param, param.replace('_', ' ').title())
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric name for display."""
        format_map = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1 Score",
            "training_time": "Training Time (s)",
            "prediction_time": "Prediction Time (s)"
        }
        return format_map.get(metric, metric.replace('_', ' ').title())
    
    def _format_experiment_name(self, exp_type: str) -> str:
        """Format experiment type name for display."""
        format_map = {
            "data_size_scaling": "Data Size",
            "tree_count_scaling": "Tree Count",
            "depth_scaling": "Max Depth"
        }
        return format_map.get(exp_type, exp_type.replace('_', ' ').title())
    
    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        """Save plot in multiple formats."""
        if self.save_dir is None:
            return
        
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for fmt in self.plot_config['save_formats']:
            filepath = save_dir / f"{filename}.{fmt}"
            fig.savefig(
                filepath,
                format=fmt,
                dpi=self.plot_config['dpi'],
                bbox_inches='tight',
                facecolor='white'
            )
            logger.debug(f"Saved plot: {filepath}")


def create_all_visualizations(
    all_results: Dict[str, List[Dict[str, Any]]],
    analysis_results: Dict[str, Dict[str, Any]],
    dataset_name: str = "adult",
    save_dir: Optional[Path] = None
) -> Dict[str, plt.Figure]:
    """
    Create all visualizations for scaling experiments.
    
    Args:
        all_results: Dictionary of experiment results
        analysis_results: Dictionary of analysis results
        dataset_name: Name of the dataset
        save_dir: Directory to save plots
        
    Returns:
        Dictionary of figure names to Figure objects
    """
    logger.info(f"Creating visualizations for {dataset_name} dataset")
    
    if save_dir is None:
        save_dir = config.get_plots_path(dataset_name)
    
    plotter = ScalingPlotter(save_dir)
    figures = {}
    
    # Create individual metric plots for each experiment
    experiment_params = {
        "data_size_scaling": "data_size",
        "tree_count_scaling": "n_estimators", 
        "depth_scaling": "max_depth"
    }
    
    for exp_type, x_param in experiment_params.items():
        if exp_type in all_results:
            exp_analysis = analysis_results.get(exp_type, {})
            exp_figures = plotter.plot_all_metrics_scaling(
                all_results[exp_type], x_param, exp_analysis,
                title_prefix=f"{dataset_name.title()} - ",
                save_prefix=f"{dataset_name}_{exp_type}_"
            )
            
            # Add to main figures dict with prefixed names
            for metric, fig in exp_figures.items():
                figures[f"{exp_type}_{metric}"] = fig
    
    # Create summary plot
    summary_fig = plotter.plot_experiment_summary(all_results, analysis_results, dataset_name)
    figures["summary"] = summary_fig
    
    # Create power law parameters plot
    params_fig = plotter.plot_power_law_parameters(analysis_results, dataset_name)
    figures["power_law_parameters"] = params_fig
    
    logger.info(f"Created {len(figures)} visualizations")
    return figures