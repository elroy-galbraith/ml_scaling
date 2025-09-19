"""Practical decision framework for Random Forest scaling experiments."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


class ScalingDecisionFramework:
    """
    Practical decision framework for Random Forest scaling experiments.

    Provides tools for cost-performance trade-off analysis, resource estimation,
    and optimal hyperparameter selection based on constraints.
    """

    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize the decision framework with experimental results.

        Args:
            results_df: DataFrame containing scaling experiment results
        """
        self.results_df = results_df.copy()
        self.scaling_laws = {}
        self.fitted_models = {}

        # Validate required columns
        required_cols = ['n_samples_train', 'training_time_mean', 'training_memory_mb_mean']
        missing_cols = [col for col in required_cols if col not in self.results_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _power_law(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Power law function: y = a * x^b"""
        return a * np.power(x, b)

    def _fit_scaling_law(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, float]:
        """
        Fit power law scaling to data.

        Args:
            x_data: Independent variable (e.g., n_samples)
            y_data: Dependent variable (e.g., training_time)

        Returns:
            Dictionary with fitted parameters and quality metrics
        """
        try:
            # Filter out invalid data
            valid_mask = (x_data > 0) & (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]

            if len(x_clean) < 3:
                return {'a': 1.0, 'b': 1.0, 'r_squared': 0.0, 'valid': False}

            # Fit power law
            popt, pcov = curve_fit(self._power_law, x_clean, y_clean, maxfev=1000)
            a, b = popt

            # Calculate R-squared
            y_pred = self._power_law(x_clean, a, b)
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                'a': a,
                'b': b,
                'r_squared': r_squared,
                'valid': True,
                'rmse': np.sqrt(np.mean((y_clean - y_pred) ** 2))
            }
        except Exception:
            return {'a': 1.0, 'b': 1.0, 'r_squared': 0.0, 'valid': False}

    def fit_scaling_laws(self) -> Dict[str, Dict[str, float]]:
        """
        Fit scaling laws for time and memory consumption.

        Returns:
            Dictionary of fitted scaling laws
        """
        # Filter data scaling experiments only
        data_scaling = self.results_df[self.results_df['scaling_type'] == 'data'].copy()

        if len(data_scaling) == 0:
            print("Warning: No data scaling experiments found")
            return {}

        scaling_laws = {}

        # Fit training time scaling law
        if 'training_time_mean' in data_scaling.columns:
            x_data = data_scaling['n_samples_train'].values
            y_data = data_scaling['training_time_mean'].values
            scaling_laws['training_time'] = self._fit_scaling_law(x_data, y_data)

        # Fit memory scaling law
        if 'training_memory_mb_mean' in data_scaling.columns:
            x_data = data_scaling['n_samples_train'].values
            y_data = data_scaling['training_memory_mb_mean'].values
            scaling_laws['memory'] = self._fit_scaling_law(x_data, y_data)

        # Fit prediction time scaling law
        if 'prediction_time_mean' in data_scaling.columns:
            x_data = data_scaling['n_samples_train'].values
            y_data = data_scaling['prediction_time_mean'].values
            scaling_laws['prediction_time'] = self._fit_scaling_law(x_data, y_data)

        self.scaling_laws = scaling_laws
        return scaling_laws

    def estimate_resources(self, n_samples: int, n_estimators: int = 100) -> Dict[str, float]:
        """
        Estimate computational resources required for given problem size.

        Args:
            n_samples: Number of training samples
            n_estimators: Number of estimators in Random Forest

        Returns:
            Dictionary with estimated resources
        """
        if not self.scaling_laws:
            self.fit_scaling_laws()

        estimates = {'n_samples': n_samples, 'n_estimators': n_estimators}

        # Base estimates from scaling laws
        for metric, law in self.scaling_laws.items():
            if law['valid']:
                base_estimate = law['a'] * (n_samples ** law['b'])

                # Scale by n_estimators (rough approximation)
                if metric in ['training_time', 'memory']:
                    estimator_scaling = n_estimators / 100.0  # Assume linear scaling
                    estimates[f'{metric}_estimated'] = base_estimate * estimator_scaling
                else:
                    estimates[f'{metric}_estimated'] = base_estimate

        # Add confidence intervals based on R-squared
        for metric, law in self.scaling_laws.items():
            if law['valid'] and f'{metric}_estimated' in estimates:
                uncertainty_factor = 1 + (1 - law['r_squared'])
                estimates[f'{metric}_uncertainty_factor'] = uncertainty_factor
                estimates[f'{metric}_lower'] = estimates[f'{metric}_estimated'] / uncertainty_factor
                estimates[f'{metric}_upper'] = estimates[f'{metric}_estimated'] * uncertainty_factor

        return estimates

    def find_optimal_parameters(
        self,
        max_time_seconds: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        min_performance: Optional[float] = None,
        n_samples: int = 10000
    ) -> Dict[str, Any]:
        """
        Find optimal Random Forest parameters given constraints.

        Args:
            max_time_seconds: Maximum acceptable training time
            max_memory_mb: Maximum acceptable memory usage
            min_performance: Minimum acceptable performance (accuracy/f1)
            n_samples: Number of training samples

        Returns:
            Dictionary with optimal parameters and expected performance
        """
        # Filter parameter scaling experiments
        param_scaling = self.results_df[self.results_df['scaling_type'] == 'parameters'].copy()

        if len(param_scaling) == 0:
            return {'error': 'No parameter scaling experiments found'}

        # Add estimated resources for given n_samples
        param_scaling['estimated_time'] = param_scaling.apply(
            lambda row: self._estimate_resource_for_row(row, n_samples, 'training_time'), axis=1
        )
        param_scaling['estimated_memory'] = param_scaling.apply(
            lambda row: self._estimate_resource_for_row(row, n_samples, 'memory'), axis=1
        )

        # Apply constraints
        candidates = param_scaling.copy()

        if max_time_seconds is not None:
            candidates = candidates[candidates['estimated_time'] <= max_time_seconds]

        if max_memory_mb is not None:
            candidates = candidates[candidates['estimated_memory'] <= max_memory_mb]

        # Performance constraint
        performance_col = None
        for col in ['accuracy_mean', 'f1_score_mean']:
            if col in candidates.columns:
                performance_col = col
                break

        if min_performance is not None and performance_col is not None:
            candidates = candidates[candidates[performance_col] >= min_performance]

        if len(candidates) == 0:
            return {'error': 'No parameter combinations satisfy all constraints'}

        # Select best performer among candidates
        if performance_col is not None:
            best_idx = candidates[performance_col].idxmax()
            best_config = candidates.loc[best_idx]

            result = {
                'optimal_params': best_config['model_params'],
                'expected_performance': best_config[performance_col],
                'estimated_time_seconds': best_config['estimated_time'],
                'estimated_memory_mb': best_config['estimated_memory'],
                'confidence': best_config.get(f"{performance_col.replace('_mean', '_std')}", 0),
                'n_candidates': len(candidates)
            }
        else:
            # If no performance metric, choose fastest
            best_idx = candidates['estimated_time'].idxmin()
            best_config = candidates.loc[best_idx]

            result = {
                'optimal_params': best_config['model_params'],
                'estimated_time_seconds': best_config['estimated_time'],
                'estimated_memory_mb': best_config['estimated_memory'],
                'n_candidates': len(candidates)
            }

        return result

    def _estimate_resource_for_row(self, row: pd.Series, n_samples: int, resource_type: str) -> float:
        """Estimate resource usage for a parameter combination at given sample size."""
        if resource_type not in self.scaling_laws or not self.scaling_laws[resource_type]['valid']:
            return row.get(f'{resource_type}_mean', 0)

        law = self.scaling_laws[resource_type]
        base_samples = row['n_samples_train']
        base_value = row.get(f'{resource_type}_mean', 0)

        if base_samples <= 0 or base_value <= 0:
            return base_value

        # Scale based on sample size difference
        scaling_factor = (n_samples / base_samples) ** law['b']

        # Scale by n_estimators if relevant
        if resource_type in ['training_time', 'memory']:
            current_estimators = row['model_params'].get('n_estimators', 100)
            estimator_scaling = current_estimators / 100.0
            return base_value * scaling_factor * estimator_scaling

        return base_value * scaling_factor

    def detect_performance_plateau(self, metric: str = 'accuracy_mean') -> Dict[str, Any]:
        """
        Detect performance plateaus to identify diminishing returns.

        Args:
            metric: Performance metric to analyze

        Returns:
            Dictionary with plateau detection results
        """
        if metric not in self.results_df.columns:
            return {'error': f'Metric {metric} not found in results'}

        # Analyze data scaling experiments
        data_scaling = self.results_df[self.results_df['scaling_type'] == 'data'].copy()
        data_scaling = data_scaling.sort_values('n_samples_train')

        if len(data_scaling) < 3:
            return {'error': 'Insufficient data for plateau detection'}

        x = data_scaling['n_samples_train'].values
        y = data_scaling[metric].values

        # Calculate marginal improvements
        marginal_improvements = np.diff(y) / np.diff(x)

        # Find where improvements drop below threshold
        improvement_threshold = np.max(marginal_improvements) * 0.1  # 10% of max improvement
        plateau_candidates = np.where(marginal_improvements < improvement_threshold)[0]

        if len(plateau_candidates) > 0:
            plateau_start_idx = plateau_candidates[0]
            plateau_start_samples = x[plateau_start_idx]
            plateau_performance = y[plateau_start_idx]
        else:
            plateau_start_samples = None
            plateau_performance = None

        # Analyze parameter scaling for diminishing returns
        param_scaling = self.results_df[self.results_df['scaling_type'] == 'parameters'].copy()
        param_by_estimators = param_scaling.groupby(
            param_scaling['model_params'].apply(lambda x: x.get('n_estimators', 100))
        )[metric].mean().sort_index()

        return {
            'data_scaling_plateau': {
                'plateau_detected': plateau_start_samples is not None,
                'plateau_start_samples': plateau_start_samples,
                'plateau_performance': plateau_performance,
                'marginal_improvements': marginal_improvements.tolist()
            },
            'parameter_scaling_plateau': {
                'estimators_vs_performance': param_by_estimators.to_dict(),
                'optimal_estimators': param_by_estimators.idxmax()
            }
        }

    def cost_performance_analysis(
        self,
        time_cost_per_second: float = 0.01,
        memory_cost_per_gb_hour: float = 0.1
    ) -> pd.DataFrame:
        """
        Perform cost-performance trade-off analysis.

        Args:
            time_cost_per_second: Cost per second of computation time
            memory_cost_per_gb_hour: Cost per GB-hour of memory usage

        Returns:
            DataFrame with cost-performance analysis
        """
        analysis_df = self.results_df.copy()

        # Calculate costs
        if 'training_time_mean' in analysis_df.columns:
            analysis_df['time_cost'] = analysis_df['training_time_mean'] * time_cost_per_second

        if 'training_memory_mb_mean' in analysis_df.columns:
            memory_gb_hours = (analysis_df['training_memory_mb_mean'] / 1024) * (analysis_df.get('training_time_mean', 0) / 3600)
            analysis_df['memory_cost'] = memory_gb_hours * memory_cost_per_gb_hour

        analysis_df['total_cost'] = analysis_df.get('time_cost', 0) + analysis_df.get('memory_cost', 0)

        # Calculate cost efficiency
        performance_col = None
        for col in ['accuracy_mean', 'f1_score_mean']:
            if col in analysis_df.columns:
                performance_col = col
                break

        if performance_col is not None:
            analysis_df['cost_efficiency'] = analysis_df[performance_col] / analysis_df['total_cost']
            analysis_df['cost_efficiency'] = analysis_df['cost_efficiency'].replace([np.inf, -np.inf], 0)

        return analysis_df[['experiment_id', 'n_samples_train', 'model_params',
                          'total_cost', performance_col, 'cost_efficiency']].dropna()

    def generate_decision_guide(self) -> Dict[str, Any]:
        """
        Generate a comprehensive decision guide based on experimental results.

        Returns:
            Dictionary containing practical recommendations
        """
        if not self.scaling_laws:
            self.fit_scaling_laws()

        # Performance plateau analysis
        plateau_analysis = self.detect_performance_plateau()

        # Sample size recommendations
        sample_recommendations = {}
        if 'data_scaling_plateau' in plateau_analysis:
            plateau_info = plateau_analysis['data_scaling_plateau']
            if plateau_info['plateau_detected']:
                sample_recommendations['optimal_samples'] = plateau_info['plateau_start_samples']
                sample_recommendations['reasoning'] = "Performance plateau detected - additional data provides diminishing returns"
            else:
                sample_recommendations['recommendation'] = "Continue increasing sample size - no plateau detected"

        # Parameter recommendations
        param_recommendations = {}
        if 'parameter_scaling_plateau' in plateau_analysis:
            param_info = plateau_analysis['parameter_scaling_plateau']
            param_recommendations['optimal_n_estimators'] = param_info['optimal_estimators']

        # Cost efficiency analysis
        cost_analysis = self.cost_performance_analysis()
        if not cost_analysis.empty and 'cost_efficiency' in cost_analysis.columns:
            most_efficient = cost_analysis.loc[cost_analysis['cost_efficiency'].idxmax()]
            param_recommendations['most_cost_efficient'] = most_efficient['model_params']

        # Resource scaling insights
        scaling_insights = {}
        for metric, law in self.scaling_laws.items():
            if law['valid']:
                if law['b'] < 1:
                    scaling_insights[metric] = "Sub-linear scaling - efficient use of resources"
                elif law['b'] > 1.5:
                    scaling_insights[metric] = "Super-linear scaling - resource usage grows rapidly"
                else:
                    scaling_insights[metric] = "Near-linear scaling - predictable resource usage"

        return {
            'sample_size_recommendations': sample_recommendations,
            'parameter_recommendations': param_recommendations,
            'scaling_insights': scaling_insights,
            'scaling_laws': self.scaling_laws,
            'summary': self._generate_summary()
        }

    def _generate_summary(self) -> str:
        """Generate a text summary of key findings."""
        summary_parts = []

        # Sample size insights
        data_scaling = self.results_df[self.results_df['scaling_type'] == 'data']
        if not data_scaling.empty:
            min_samples = data_scaling['n_samples_train'].min()
            max_samples = data_scaling['n_samples_train'].max()
            summary_parts.append(f"Tested sample sizes from {min_samples:,} to {max_samples:,}")

        # Performance insights
        performance_cols = [col for col in self.results_df.columns if 'accuracy_mean' in col or 'f1_score_mean' in col]
        if performance_cols:
            best_perf = self.results_df[performance_cols[0]].max()
            summary_parts.append(f"Best performance achieved: {best_perf:.3f}")

        # Scaling law insights
        if self.scaling_laws:
            for metric, law in self.scaling_laws.items():
                if law['valid'] and law['r_squared'] > 0.8:
                    summary_parts.append(f"{metric} scales as N^{law['b']:.2f} (R²={law['r_squared']:.3f})")

        return ". ".join(summary_parts) + "."


def load_decision_framework(results_path: str) -> ScalingDecisionFramework:
    """
    Convenience function to load decision framework from results file.

    Args:
        results_path: Path to CSV file with experimental results

    Returns:
        Initialized ScalingDecisionFramework
    """
    results_df = pd.read_csv(results_path)
    return ScalingDecisionFramework(results_df)


if __name__ == "__main__":
    # Example usage
    print("Scaling Decision Framework - Example Usage")
    print("This module provides tools for:")
    print("1. Resource estimation based on scaling laws")
    print("2. Optimal parameter selection given constraints")
    print("3. Performance plateau detection")
    print("4. Cost-performance trade-off analysis")
    print("5. Practical decision recommendations")