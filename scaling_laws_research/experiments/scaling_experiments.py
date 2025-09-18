"""Core scaling experiment implementation for Random Forest models."""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import os
import pickle

from ..utils.data_generator import DataGenerator
from ..utils.performance_tracker import PerformanceTracker, PerformanceMetrics
from ..utils.config import ExperimentConfig


class ScalingExperiment:
    """Conducts Random Forest scaling experiments."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the scaling experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.data_generator = DataGenerator(random_state=config.data.random_state)
        self.performance_tracker = PerformanceTracker()
        self.results = []
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
    def _create_model(self, task_type: str, **params) -> Any:
        """
        Create a Random Forest model based on task type.
        
        Args:
            task_type: Either "classification" or "regression"
            **params: Model parameters
            
        Returns:
            Random Forest model instance
        """
        if task_type == "classification":
            return RandomForestClassifier(**params)
        elif task_type == "regression":
            return RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
    
    def _evaluate_model(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        task_type: str
    ) -> float:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            task_type: Either "classification" or "regression"
            
        Returns:
            Performance score
        """
        predictions = model.predict(X_test)
        
        if task_type == "classification":
            return accuracy_score(y_test, predictions)
        else:  # regression
            return mean_squared_error(y_test, predictions)
    
    def run_single_experiment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Dict[str, Any],
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Run a single scaling experiment.
        
        Args:
            X: Features
            y: Labels/targets
            model_params: Random Forest parameters
            experiment_id: Unique identifier for this experiment
            
        Returns:
            Dictionary containing experiment results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.data.test_size,
            random_state=self.config.data.random_state
        )
        
        # Create model
        model = self._create_model(self.config.data.task_type, **model_params)
        
        # Track training performance
        self.performance_tracker.reset()
        training_time = self.performance_tracker.track_training(model, X_train, y_train)
        training_memory = self.performance_tracker.get_peak_memory()
        training_cpu = self.performance_tracker.get_avg_cpu()
        
        # Track prediction performance
        self.performance_tracker.reset()
        predictions, prediction_time = self.performance_tracker.track_prediction(model, X_test)
        prediction_memory = self.performance_tracker.get_peak_memory()
        prediction_cpu = self.performance_tracker.get_avg_cpu()
        
        # Evaluate performance
        if self.config.data.task_type == "classification":
            score = accuracy_score(y_test, predictions)
            score_name = "accuracy"
        else:
            score = mean_squared_error(y_test, predictions)
            score_name = "mse"
        
        # Compile results
        result = {
            'experiment_id': experiment_id,
            'timestamp': time.time(),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'n_features': X.shape[1],
            'task_type': self.config.data.task_type,
            'model_params': model_params,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'training_memory_mb': training_memory,
            'prediction_memory_mb': prediction_memory,
            'training_cpu_percent': training_cpu,
            'prediction_cpu_percent': prediction_cpu,
            score_name: score
        }
        
        # Save model if requested
        if self.config.save_models:
            model_path = os.path.join(self.config.output_dir, f"model_{experiment_id}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            result['model_path'] = model_path
        
        # Save predictions if requested
        if self.config.save_predictions:
            pred_path = os.path.join(self.config.output_dir, f"predictions_{experiment_id}.npy")
            np.save(pred_path, predictions)
            result['predictions_path'] = pred_path
        
        if self.config.verbose:
            print(f"Completed experiment {experiment_id}: "
                  f"Training time: {training_time:.3f}s, "
                  f"{score_name}: {score:.4f}")
        
        return result
    
    def run_data_scaling_experiment(self) -> List[Dict[str, Any]]:
        """
        Run experiments with different dataset sizes.
        
        Returns:
            List of experiment results
        """
        results = []
        
        # Generate datasets with different scales
        datasets = self.data_generator.generate_scaling_datasets(
            base_samples=self.config.data.base_samples,
            base_features=self.config.data.base_features,
            sample_scales=self.config.data.sample_scales,
            feature_scales=self.config.data.feature_scales,
            task_type=self.config.data.task_type
        )
        
        # Use default Random Forest parameters for data scaling
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': self.config.random_forest.random_state,
            'n_jobs': self.config.random_forest.n_jobs
        }
        
        for dataset_name, (X, y) in datasets.items():
            experiment_id = f"data_scaling_{dataset_name}"
            result = self.run_single_experiment(X, y, default_params, experiment_id)
            result['scaling_type'] = 'data'
            result['dataset_name'] = dataset_name
            results.append(result)
        
        return results
    
    def run_parameter_scaling_experiment(self) -> List[Dict[str, Any]]:
        """
        Run experiments with different Random Forest parameters.
        
        Returns:
            List of experiment results
        """
        results = []
        
        # Generate a single dataset
        X, y = self.data_generator.generate_classification_dataset(
            n_samples=self.config.data.base_samples,
            n_features=self.config.data.base_features
        ) if self.config.data.task_type == "classification" else \
            self.data_generator.generate_regression_dataset(
                n_samples=self.config.data.base_samples,
                n_features=self.config.data.base_features
            )
        
        # Get parameter grid
        param_grid = self.config.get_parameter_grid()
        
        for i, params in enumerate(param_grid):
            experiment_id = f"param_scaling_{i:04d}"
            result = self.run_single_experiment(X, y, params, experiment_id)
            result['scaling_type'] = 'parameters'
            result['param_combination'] = i
            results.append(result)
        
        return results
    
    def run_full_experiment(self) -> pd.DataFrame:
        """
        Run complete scaling experiment including both data and parameter scaling.
        
        Returns:
            DataFrame containing all results
        """
        if self.config.verbose:
            print("Starting Random Forest scaling experiment...")
            print(f"Configuration: {self.config.name}")
            print(f"Task type: {self.config.data.task_type}")
            print(f"Output directory: {self.config.output_dir}")
        
        all_results = []
        
        # Run data scaling experiments
        if self.config.verbose:
            print("\nRunning data scaling experiments...")
        data_results = self.run_data_scaling_experiment()
        all_results.extend(data_results)
        
        # Run parameter scaling experiments
        if self.config.verbose:
            print("\nRunning parameter scaling experiments...")
        param_results = self.run_parameter_scaling_experiment()
        all_results.extend(param_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        results_path = os.path.join(self.config.output_dir, "scaling_results.csv")
        results_df.to_csv(results_path, index=False)
        
        # Save configuration
        config_path = os.path.join(self.config.output_dir, "experiment_config.json")
        self.config.save_to_file(config_path)
        
        if self.config.verbose:
            print(f"\nExperiment completed!")
            print(f"Results saved to: {results_path}")
            print(f"Configuration saved to: {config_path}")
            print(f"Total experiments: {len(results_df)}")
        
        self.results = all_results
        return results_df