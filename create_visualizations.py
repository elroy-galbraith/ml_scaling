#!/usr/bin/env python3
"""Create visualizations for both Adult and synthetic dataset results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def fix_metric_name(metric, df_columns):
    """Fix metric name to match aggregated column names."""
    if metric in df_columns:
        return metric

    # Try adding _mean suffix
    if f"{metric}_mean" in df_columns:
        return f"{metric}_mean"

    # Try other variations
    for suffix in ['_mean', '_median', '']:
        test_name = f"{metric}{suffix}"
        if test_name in df_columns:
            return test_name

    raise ValueError(f"Metric '{metric}' not found. Available columns: {list(df_columns)}")

def create_scaling_plot(results_df, metric, title_prefix="", save_path=None):
    """Create a scaling plot for the given metric."""
    # Filter data scaling results
    data_results = results_df[results_df['scaling_type'] == 'data'].copy()

    if data_results.empty:
        print(f"No data scaling results found for {title_prefix}")
        return None

    # Fix metric name
    metric_col = fix_metric_name(metric, data_results.columns)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Plot vs number of samples
    ax1 = axes[0]
    sns.scatterplot(data=data_results, x='n_samples_train', y=metric_col, ax=ax1, s=100, alpha=0.7)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Training Samples')
    ax1.set_ylabel(metric.replace('_', ' ').title())
    ax1.set_title(f'{title_prefix}{metric.replace("_", " ").title()} vs Sample Size')
    ax1.grid(True, alpha=0.3)

    # Plot vs number of features
    ax2 = axes[1]
    sns.scatterplot(data=data_results, x='n_features', y=metric_col, ax=ax2, s=100, alpha=0.7)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel(metric.replace('_', ' ').title())
    ax2.set_title(f'{title_prefix}{metric.replace("_", " ").title()} vs Feature Count')
    ax2.grid(True, alpha=0.3)

    # Dataset complexity plot
    ax3 = axes[2]
    data_results['dataset_complexity'] = data_results['n_samples_train'] * data_results['n_features']
    sns.scatterplot(data=data_results, x='dataset_complexity', y=metric_col, ax=ax3, s=100, alpha=0.7)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Dataset Complexity (samples × features)')
    ax3.set_ylabel(metric.replace('_', ' ').title())
    ax3.set_title(f'{title_prefix}{metric.replace("_", " ").title()} vs Dataset Complexity')
    ax3.grid(True, alpha=0.3)

    # Parameter scaling comparison
    ax4 = axes[3]
    param_results = results_df[results_df['scaling_type'] == 'parameters'].copy()
    if not param_results.empty:
        metric_col_param = fix_metric_name(metric, param_results.columns)
        # Extract n_estimators from model_params for x-axis
        param_results['n_estimators'] = param_results['model_params'].apply(
            lambda x: eval(x)['n_estimators'] if isinstance(x, str) else x['n_estimators']
        )
        sns.boxplot(data=param_results, x='n_estimators', y=metric_col_param, ax=ax4)
        ax4.set_yscale('log')
        ax4.set_xlabel('Number of Estimators')
        ax4.set_ylabel(metric.replace('_', ' ').title())
        ax4.set_title(f'{title_prefix}{metric.replace("_", " ").title()} vs N Estimators')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig

def main():
    """Create visualizations for both datasets."""

    # Load results
    adult_results = pd.read_csv("results/adult_experiment_small/scaling_results.csv")
    synthetic_results = pd.read_csv("results/synthetic_experiment_small/scaling_results.csv")

    print("Adult dataset results shape:", adult_results.shape)
    print("Synthetic dataset results shape:", synthetic_results.shape)

    # Create output directory
    Path("results/visualizations").mkdir(exist_ok=True)

    # Metrics to plot
    metrics = ['training_time', 'accuracy', 'precision', 'f1_score']

    for metric in metrics:
        print(f"\nCreating plots for {metric}...")

        # Adult dataset plots
        try:
            fig_adult = create_scaling_plot(
                adult_results,
                metric,
                title_prefix="Adult Dataset: ",
                save_path=f"results/visualizations/adult_{metric}_scaling.png"
            )
            if fig_adult:
                plt.show()
                plt.close(fig_adult)
        except Exception as e:
            print(f"Error creating Adult plot for {metric}: {e}")

        # Synthetic dataset plots
        try:
            fig_synthetic = create_scaling_plot(
                synthetic_results,
                metric,
                title_prefix="Synthetic Dataset: ",
                save_path=f"results/visualizations/synthetic_{metric}_scaling.png"
            )
            if fig_synthetic:
                plt.show()
                plt.close(fig_synthetic)
        except Exception as e:
            print(f"Error creating Synthetic plot for {metric}: {e}")

    # Comparison plot
    print("\nCreating comparison plot...")
    create_comparison_plot(adult_results, synthetic_results)

def create_comparison_plot(adult_results, synthetic_results):
    """Create side-by-side comparison of Adult vs Synthetic results."""

    # Filter data scaling results
    adult_data = adult_results[adult_results['scaling_type'] == 'data'].copy()
    synthetic_data = synthetic_results[synthetic_results['scaling_type'] == 'data'].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Training time comparison
    ax1 = axes[0, 0]
    if not adult_data.empty:
        sns.scatterplot(data=adult_data, x='n_samples_train', y='training_time_mean',
                       ax=ax1, label='Adult Dataset', s=100, alpha=0.7)
    if not synthetic_data.empty:
        sns.scatterplot(data=synthetic_data, x='n_samples_train', y='training_time_mean',
                       ax=ax1, label='Synthetic Dataset', s=100, alpha=0.7)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Training Samples')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Scaling Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy comparison
    ax2 = axes[0, 1]
    if not adult_data.empty:
        sns.scatterplot(data=adult_data, x='n_samples_train', y='accuracy_mean',
                       ax=ax2, label='Adult Dataset', s=100, alpha=0.7)
    if not synthetic_data.empty:
        sns.scatterplot(data=synthetic_data, x='n_samples_train', y='accuracy_mean',
                       ax=ax2, label='Synthetic Dataset', s=100, alpha=0.7)
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Training Samples')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Scaling Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Feature scaling comparison (synthetic only, since Adult has fixed features)
    ax3 = axes[1, 0]
    if not synthetic_data.empty:
        sns.scatterplot(data=synthetic_data, x='n_features', y='training_time_mean',
                       ax=ax3, label='Synthetic Dataset', s=100, alpha=0.7)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title('Feature Scaling (Synthetic Only)')
        ax3.grid(True, alpha=0.3)

    # Performance distribution comparison
    ax4 = axes[1, 1]
    if not adult_data.empty and not synthetic_data.empty:
        # Create comparison data
        comparison_data = []
        for _, row in adult_data.iterrows():
            comparison_data.append({'Dataset': 'Adult', 'Accuracy': row['accuracy_mean']})
        for _, row in synthetic_data.iterrows():
            comparison_data.append({'Dataset': 'Synthetic', 'Accuracy': row['accuracy_mean']})

        comparison_df = pd.DataFrame(comparison_data)
        sns.boxplot(data=comparison_df, x='Dataset', y='Accuracy', ax=ax4)
        ax4.set_title('Accuracy Distribution Comparison')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/visualizations/dataset_comparison.png", dpi=300, bbox_inches='tight')
    print("Saved comparison plot to results/visualizations/dataset_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()