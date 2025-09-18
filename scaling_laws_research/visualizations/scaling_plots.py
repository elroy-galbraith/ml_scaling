"""Visualization tools for Random Forest scaling analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import os


class ScalingPlotter:
    """Creates various plots for analyzing Random Forest scaling behavior."""
    
    def __init__(self, style: str = "whitegrid", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the plotter.
        
        Args:
            style: Seaborn style
            figsize: Default figure size
        """
        sns.set_style(style)
        self.figsize = figsize
        plt.rcParams['figure.figsize'] = figsize
        
    def plot_data_scaling(
        self, 
        results_df: pd.DataFrame, 
        metric: str = "training_time",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot performance metrics vs dataset size.
        
        Args:
            results_df: Results DataFrame from scaling experiments
            metric: Metric to plot (training_time, prediction_time, accuracy, mse, etc.)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Filter data scaling results
        data_results = results_df[results_df['scaling_type'] == 'data'].copy()
        
        if data_results.empty:
            raise ValueError("No data scaling results found")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot vs number of samples
        ax1 = axes[0]
        sns.scatterplot(data=data_results, x='n_samples_train', y=metric, ax=ax1)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Number of Training Samples')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title(f'{metric.replace("_", " ").title()} vs Sample Size')
        ax1.grid(True, alpha=0.3)
        
        # Plot vs number of features
        ax2 = axes[1]
        sns.scatterplot(data=data_results, x='n_features', y=metric, ax=ax2)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel(metric.replace('_', ' ').title())
        ax2.set_title(f'{metric.replace("_", " ").title()} vs Feature Count')
        ax2.grid(True, alpha=0.3)
        
        # Heatmap: samples vs features
        ax3 = axes[2]
        pivot_data = data_results.pivot_table(
            values=metric, 
            index='n_features', 
            columns='n_samples_train', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', ax=ax3, cmap='viridis')
        ax3.set_title(f'{metric.replace("_", " ").title()} Heatmap')
        ax3.set_xlabel('Number of Training Samples')
        ax3.set_ylabel('Number of Features')
        
        # Combined scaling plot
        ax4 = axes[3]
        data_results['dataset_complexity'] = data_results['n_samples_train'] * data_results['n_features']
        sns.scatterplot(data=data_results, x='dataset_complexity', y=metric, ax=ax4)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_xlabel('Dataset Complexity (samples × features)')
        ax4.set_ylabel(metric.replace('_', ' ').title())
        ax4.set_title(f'{metric.replace("_", " ").title()} vs Dataset Complexity')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_parameter_scaling(
        self, 
        results_df: pd.DataFrame, 
        metric: str = "training_time",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot performance metrics vs Random Forest parameters.
        
        Args:
            results_df: Results DataFrame from scaling experiments
            metric: Metric to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Filter parameter scaling results
        param_results = results_df[results_df['scaling_type'] == 'parameters'].copy()
        
        if param_results.empty:
            raise ValueError("No parameter scaling results found")
        
        # Extract parameter values from model_params column
        param_cols = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        for col in param_cols:
            param_results[col] = param_results['model_params'].apply(lambda x: x.get(col))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, param in enumerate(param_cols):
            ax = axes[i]
            
            # Handle None values in max_depth
            if param == 'max_depth':
                plot_data = param_results[param_results[param].notna()]
                if not plot_data.empty:
                    sns.scatterplot(data=plot_data, x=param, y=metric, ax=ax)
                # Also show None values separately
                none_data = param_results[param_results[param].isna()]
                if not none_data.empty:
                    ax.scatter([0], [none_data[metric].mean()], 
                             color='red', s=100, alpha=0.7, label='None (unlimited)')
                    ax.legend()
            else:
                sns.scatterplot(data=param_results, x=param, y=metric, ax=ax)
            
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs {param.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
            # Set log scale for n_estimators
            if param == 'n_estimators':
                ax.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_scaling_laws(
        self, 
        results_df: pd.DataFrame, 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot scaling laws with power-law fits.
        
        Args:
            results_df: Results DataFrame from scaling experiments
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Filter data scaling results
        data_results = results_df[results_df['scaling_type'] == 'data'].copy()
        
        if data_results.empty:
            raise ValueError("No data scaling results found")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = ['training_time', 'prediction_time', 'training_memory_mb']
        x_vars = ['n_samples_train', 'n_features']
        
        for i, x_var in enumerate(x_vars):
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                
                # Plot data points
                x_data = data_results[x_var]
                y_data = data_results[metric]
                
                # Remove any zero or negative values for log-log plot
                valid_mask = (x_data > 0) & (y_data > 0)
                x_clean = x_data[valid_mask]
                y_clean = y_data[valid_mask]
                
                if len(x_clean) < 2:
                    ax.text(0.5, 0.5, 'Insufficient data', 
                           transform=ax.transAxes, ha='center', va='center')
                    continue
                
                # Check for unique values
                if len(np.unique(x_clean)) < 2:
                    ax.text(0.5, 0.5, 'No variation in x', 
                           transform=ax.transAxes, ha='center', va='center')
                    continue
                
                ax.scatter(x_clean, y_clean, alpha=0.7, s=50)
                
                # Fit power law: y = a * x^b
                log_x = np.log(x_clean)
                log_y = np.log(y_clean)
                coeffs = np.polyfit(log_x, log_y, 1)
                b, log_a = coeffs
                a = np.exp(log_a)
                
                # Plot fit line
                x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_fit = a * (x_fit ** b)
                ax.plot(x_fit, y_fit, 'r--', alpha=0.8, 
                       label=f'y = {a:.2e} × x^{b:.2f}')
                
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel(x_var.replace('_', ' ').title())
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} vs {x_var.replace("_", " ").title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Summary statistics in the last subplot
        ax_summary = axes[1, 2]
        ax_summary.axis('off')
        
        # Calculate R-squared for fits
        summary_text = "Scaling Law Summary\n\n"
        for x_var in x_vars:
            for metric in metrics:
                x_data = data_results[x_var]
                y_data = data_results[metric]
                valid_mask = (x_data > 0) & (y_data > 0)
                x_clean = x_data[valid_mask]
                y_clean = y_data[valid_mask]
                
                if len(x_clean) >= 2:
                    log_x = np.log(x_clean)
                    log_y = np.log(y_clean)
                    coeffs = np.polyfit(log_x, log_y, 1)
                    b = coeffs[0]
                    
                    # Calculate R-squared
                    y_pred = np.polyval(coeffs, log_x)
                    ss_res = np.sum((log_y - y_pred) ** 2)
                    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    summary_text += f"{metric} ~ {x_var}^{b:.2f} (R² = {r_squared:.3f})\n"
        
        ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_performance_comparison(
        self, 
        results_df: pd.DataFrame, 
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare different performance metrics.
        
        Args:
            results_df: Results DataFrame from scaling experiments
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Training time vs accuracy/mse
        ax1 = axes[0]
        if 'accuracy' in results_df.columns:
            sns.scatterplot(data=results_df, x='training_time', y='accuracy', 
                           hue='scaling_type', ax=ax1)
            ax1.set_ylabel('Accuracy')
        elif 'mse' in results_df.columns:
            sns.scatterplot(data=results_df, x='training_time', y='mse', 
                           hue='scaling_type', ax=ax1)
            ax1.set_ylabel('MSE')
            ax1.set_yscale('log')
        
        ax1.set_xlabel('Training Time (s)')
        ax1.set_xscale('log')
        ax1.set_title('Performance vs Training Time')
        ax1.grid(True, alpha=0.3)
        
        # Memory vs time
        ax2 = axes[1]
        sns.scatterplot(data=results_df, x='training_memory_mb', y='training_time', 
                       hue='scaling_type', ax=ax2)
        ax2.set_xlabel('Training Memory (MB)')
        ax2.set_ylabel('Training Time (s)')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_title('Training Time vs Memory Usage')
        ax2.grid(True, alpha=0.3)
        
        # Training vs prediction time
        ax3 = axes[2]
        sns.scatterplot(data=results_df, x='training_time', y='prediction_time', 
                       hue='scaling_type', ax=ax3)
        ax3.set_xlabel('Training Time (s)')
        ax3.set_ylabel('Prediction Time (s)')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_title('Training vs Prediction Time')
        ax3.grid(True, alpha=0.3)
        
        # CPU usage comparison
        ax4 = axes[3]
        cpu_data = results_df[['training_cpu_percent', 'prediction_cpu_percent', 'scaling_type']].melt(
            id_vars=['scaling_type'], 
            var_name='phase', 
            value_name='cpu_percent'
        )
        sns.boxplot(data=cpu_data, x='phase', y='cpu_percent', hue='scaling_type', ax=ax4)
        ax4.set_xlabel('Phase')
        ax4.set_ylabel('CPU Usage (%)')
        ax4.set_title('CPU Usage by Phase')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig