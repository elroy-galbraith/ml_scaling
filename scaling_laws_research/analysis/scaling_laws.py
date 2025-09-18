"""Analysis tools for extracting and interpreting Random Forest scaling laws."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.metrics import r2_score
import warnings


class ScalingLawAnalyzer:
    """Analyzes Random Forest scaling behavior and extracts scaling laws."""
    
    def __init__(self):
        """Initialize the scaling law analyzer."""
        pass
    
    def fit_power_law(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        min_points: int = 3
    ) -> Dict[str, float]:
        """
        Fit a power law of the form y = a * x^b to the data.
        
        Args:
            x: Independent variable
            y: Dependent variable
            min_points: Minimum number of points required for fitting
            
        Returns:
            Dictionary with fit parameters and statistics
        """
        # Remove invalid values
        valid_mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) < min_points:
            return {
                'a': np.nan, 'b': np.nan, 'r_squared': np.nan, 
                'p_value': np.nan, 'n_points': len(x_clean),
                'valid': False
            }
        
        # Check for unique x values
        if len(np.unique(x_clean)) < 2:
            return {
                'a': np.nan, 'b': np.nan, 'r_squared': np.nan, 
                'p_value': np.nan, 'n_points': len(x_clean),
                'valid': False
            }
        
        # Fit in log space: log(y) = log(a) + b * log(x)
        log_x = np.log(x_clean)
        log_y = np.log(y_clean)
        
        # Check for unique log x values
        if len(np.unique(log_x)) < 2:
            return {
                'a': np.nan, 'b': np.nan, 'r_squared': np.nan, 
                'p_value': np.nan, 'n_points': len(x_clean),
                'valid': False
            }
        
        # Linear regression in log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        
        # Convert back to power law parameters
        a = np.exp(intercept)
        b = slope
        r_squared = r_value ** 2
        
        return {
            'a': a,
            'b': b,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': std_err,
            'n_points': len(x_clean),
            'valid': True
        }
    
    def analyze_data_scaling(self, results_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze scaling laws for different data dimensions.
        
        Args:
            results_df: Results DataFrame from scaling experiments
            
        Returns:
            Dictionary with scaling analysis results
        """
        # Filter data scaling results
        data_results = results_df[results_df['scaling_type'] == 'data'].copy()
        
        if data_results.empty:
            raise ValueError("No data scaling results found")
        
        analysis = {}
        
        # Metrics to analyze
        metrics = ['training_time', 'prediction_time', 'training_memory_mb', 'prediction_memory_mb']
        
        # Independent variables
        x_vars = ['n_samples_train', 'n_features']
        
        for x_var in x_vars:
            analysis[x_var] = {}
            
            for metric in metrics:
                if metric not in data_results.columns:
                    continue
                    
                x_data = data_results[x_var].values
                y_data = data_results[metric].values
                
                # Fit power law
                fit_result = self.fit_power_law(x_data, y_data)
                analysis[x_var][metric] = fit_result
                
                # Add interpretation
                if fit_result['valid'] and fit_result['r_squared'] > 0.7:
                    if fit_result['b'] < 1:
                        complexity = "sub-linear"
                    elif fit_result['b'] > 1:
                        complexity = "super-linear"
                    else:
                        complexity = "linear"
                    
                    analysis[x_var][metric]['complexity'] = complexity
                    analysis[x_var][metric]['interpretation'] = self._interpret_scaling(
                        metric, x_var, fit_result['b'], complexity
                    )
        
        return analysis
    
    def analyze_parameter_scaling(self, results_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze scaling laws for Random Forest parameters.
        
        Args:
            results_df: Results DataFrame from scaling experiments
            
        Returns:
            Dictionary with parameter scaling analysis
        """
        # Filter parameter scaling results
        param_results = results_df[results_df['scaling_type'] == 'parameters'].copy()
        
        if param_results.empty:
            raise ValueError("No parameter scaling results found")
        
        # Extract parameter values
        param_cols = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        for col in param_cols:
            param_results[col] = param_results['model_params'].apply(lambda x: x.get(col))
        
        analysis = {}
        metrics = ['training_time', 'prediction_time', 'training_memory_mb']
        
        for param in param_cols:
            analysis[param] = {}
            
            # Handle None values in max_depth
            if param == 'max_depth':
                param_data = param_results[param_results[param].notna()]
            else:
                param_data = param_results
            
            if param_data.empty:
                continue
            
            for metric in metrics:
                if metric not in param_data.columns:
                    continue
                
                x_data = param_data[param].values
                y_data = param_data[metric].values
                
                # Fit power law
                fit_result = self.fit_power_law(x_data, y_data)
                analysis[param][metric] = fit_result
                
                # Add parameter-specific interpretation
                if fit_result['valid']:
                    analysis[param][metric]['interpretation'] = self._interpret_parameter_scaling(
                        metric, param, fit_result['b']
                    )
        
        return analysis
    
    def _interpret_scaling(
        self, 
        metric: str, 
        x_var: str, 
        exponent: float, 
        complexity: str
    ) -> str:
        """Generate human-readable interpretation of scaling behavior."""
        
        base_interpretations = {
            'training_time': {
                'n_samples_train': f"Training time scales as O(n^{exponent:.2f}) with sample size",
                'n_features': f"Training time scales as O(d^{exponent:.2f}) with feature count"
            },
            'prediction_time': {
                'n_samples_train': f"Prediction time scales as O(n^{exponent:.2f}) with training sample size",
                'n_features': f"Prediction time scales as O(d^{exponent:.2f}) with feature count"
            },
            'training_memory_mb': {
                'n_samples_train': f"Memory usage scales as O(n^{exponent:.2f}) with sample size",
                'n_features': f"Memory usage scales as O(d^{exponent:.2f}) with feature count"
            }
        }
        
        base_msg = base_interpretations.get(metric, {}).get(x_var, 
                                                           f"{metric} scales as {x_var}^{exponent:.2f}")
        
        if complexity == "sub-linear":
            efficiency_msg = " (efficient scaling)"
        elif complexity == "super-linear":
            efficiency_msg = " (expensive scaling)"
        else:
            efficiency_msg = " (linear scaling)"
        
        return base_msg + efficiency_msg
    
    def _interpret_parameter_scaling(self, metric: str, param: str, exponent: float) -> str:
        """Generate interpretation for parameter scaling."""
        
        param_effects = {
            'n_estimators': {
                'training_time': f"Training time increases as O(trees^{exponent:.2f})",
                'prediction_time': f"Prediction time increases as O(trees^{exponent:.2f})",
                'training_memory_mb': f"Memory usage increases as O(trees^{exponent:.2f})"
            },
            'max_depth': {
                'training_time': f"Training time scales as O(depth^{exponent:.2f})",
                'prediction_time': f"Prediction time scales as O(depth^{exponent:.2f})",
                'training_memory_mb': f"Memory usage scales as O(depth^{exponent:.2f})"
            }
        }
        
        return param_effects.get(param, {}).get(metric, 
                                               f"{metric} scales as {param}^{exponent:.2f}")
    
    def generate_summary_report(
        self, 
        data_analysis: Dict[str, Dict[str, Any]], 
        param_analysis: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate a summary report of scaling analysis.
        
        Args:
            data_analysis: Results from analyze_data_scaling
            param_analysis: Results from analyze_parameter_scaling
            
        Returns:
            Formatted summary report
        """
        report = []
        report.append("=" * 80)
        report.append("RANDOM FOREST SCALING LAWS ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Data scaling summary
        report.append("DATA SCALING ANALYSIS")
        report.append("-" * 40)
        report.append("")
        
        for x_var, metrics in data_analysis.items():
            report.append(f"{x_var.replace('_', ' ').title()}:")
            
            for metric, analysis in metrics.items():
                if not analysis.get('valid', False):
                    continue
                    
                r2 = analysis['r_squared']
                exponent = analysis['b']
                
                if r2 > 0.7:
                    quality = "Strong"
                elif r2 > 0.5:
                    quality = "Moderate"
                else:
                    quality = "Weak"
                
                report.append(f"  • {metric}: O(x^{exponent:.2f}) - {quality} fit (R² = {r2:.3f})")
                
                if 'interpretation' in analysis:
                    report.append(f"    {analysis['interpretation']}")
            
            report.append("")
        
        # Parameter scaling summary
        report.append("PARAMETER SCALING ANALYSIS")
        report.append("-" * 40)
        report.append("")
        
        for param, metrics in param_analysis.items():
            report.append(f"{param.replace('_', ' ').title()}:")
            
            for metric, analysis in metrics.items():
                if not analysis.get('valid', False):
                    continue
                    
                r2 = analysis['r_squared']
                exponent = analysis['b']
                
                if r2 > 0.7:
                    quality = "Strong"
                elif r2 > 0.5:
                    quality = "Moderate"
                else:
                    quality = "Weak"
                
                report.append(f"  • {metric}: O(param^{exponent:.2f}) - {quality} fit (R² = {r2:.3f})")
                
                if 'interpretation' in analysis:
                    report.append(f"    {analysis['interpretation']}")
            
            report.append("")
        
        # Key insights
        report.append("KEY INSIGHTS")
        report.append("-" * 40)
        report.append("")
        
        insights = self._extract_key_insights(data_analysis, param_analysis)
        for insight in insights:
            report.append(f"• {insight}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _extract_key_insights(
        self, 
        data_analysis: Dict[str, Dict[str, Any]], 
        param_analysis: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Extract key insights from scaling analysis."""
        insights = []
        
        # Check for concerning scaling patterns
        for x_var, metrics in data_analysis.items():
            for metric, analysis in metrics.items():
                if not analysis.get('valid', False):
                    continue
                
                exponent = analysis['b']
                r2 = analysis['r_squared']
                
                if r2 > 0.7:  # Only comment on strong fits
                    if metric == 'training_time' and exponent > 1.5:
                        insights.append(f"Training time shows super-linear scaling with {x_var} (O(n^{exponent:.2f})) - consider data preprocessing")
                    elif metric == 'training_memory_mb' and exponent > 1.2:
                        insights.append(f"Memory usage grows faster than linearly with {x_var} (O(n^{exponent:.2f})) - memory may become limiting")
        
        # Parameter insights
        for param, metrics in param_analysis.items():
            training_time_analysis = metrics.get('training_time', {})
            if training_time_analysis.get('valid', False) and training_time_analysis['r_squared'] > 0.7:
                exponent = training_time_analysis['b']
                if param == 'n_estimators' and exponent > 1.1:
                    insights.append(f"Training time scales super-linearly with {param} (O(n^{exponent:.2f})) - diminishing returns likely")
        
        if not insights:
            insights.append("Scaling patterns appear reasonable for Random Forest models")
        
        return insights