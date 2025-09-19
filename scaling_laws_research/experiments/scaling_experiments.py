"""Core scaling experiment implementation for Random Forest models."""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, roc_auc_score
import os
import pickle

from ..utils.data_generator import DataGenerator
from ..utils.performance_tracker import PerformanceTracker, PerformanceMetrics
from ..utils.config import ExperimentConfig
from ..data.adult_dataset import AdultDatasetLoader


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

        # Initialize dataset loader if using real data
        if config.data.dataset_name == "adult":
            self.adult_loader = AdultDatasetLoader(data_dir="data/raw")
        else:
            self.adult_loader = None

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
    ) -> Dict[str, float]:
        """
        Evaluate model performance with extended metrics.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            task_type: Either "classification" or "regression"

        Returns:
            Dictionary of performance metrics
        """
        predictions = model.predict(X_test)

        if task_type == "classification":
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
            }

            # Add ROC-AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                except Exception:
                    metrics['roc_auc'] = 0.0
            else:
                metrics['roc_auc'] = 0.0

            return metrics
        else:  # regression
            return {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions))
            }
    
    def run_single_experiment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Dict[str, Any],
        experiment_id: str,
        random_seed: int = None
    ) -> Dict[str, Any]:
        """
        Run a single scaling experiment with a specific random seed.

        Args:
            X: Features
            y: Labels/targets
            model_params: Random Forest parameters
            experiment_id: Unique identifier for this experiment
            random_seed: Random seed for this specific run

        Returns:
            Dictionary containing experiment results
        """
        if random_seed is None:
            random_seed = self.config.data.random_state

        # Split data with specific seed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.data.test_size,
            random_state=random_seed
        )

        # Create model with specific seed
        model_params_with_seed = model_params.copy()
        model_params_with_seed['random_state'] = random_seed
        model = self._create_model(self.config.data.task_type, **model_params_with_seed)

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

        # Evaluate performance with extended metrics
        performance_metrics = self._evaluate_model(model, X_test, y_test, self.config.data.task_type)

        # Compile results
        result = {
            'experiment_id': experiment_id,
            'random_seed': random_seed,
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
        }

        # Add all performance metrics
        result.update(performance_metrics)

        # Save model if requested
        if self.config.save_models:
            model_path = os.path.join(self.config.output_dir, f"model_{experiment_id}_seed_{random_seed}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            result['model_path'] = model_path

        # Save predictions if requested
        if self.config.save_predictions:
            pred_path = os.path.join(self.config.output_dir, f"predictions_{experiment_id}_seed_{random_seed}.npy")
            np.save(pred_path, predictions)
            result['predictions_path'] = pred_path

        if self.config.verbose:
            primary_metric = performance_metrics.get('accuracy', performance_metrics.get('mse', 0))
            primary_metric_name = 'accuracy' if 'accuracy' in performance_metrics else 'mse'
            print(f"Completed experiment {experiment_id} (seed {random_seed}): "
                  f"Training time: {training_time:.3f}s, "
                  f"{primary_metric_name}: {primary_metric:.4f}")

        return result

    def run_multi_seed_experiment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_params: Dict[str, Any],
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Run experiment with multiple random seeds and aggregate results.

        Args:
            X: Features
            y: Labels/targets
            model_params: Random Forest parameters
            experiment_id: Unique identifier for this experiment

        Returns:
            Dictionary containing aggregated results across seeds
        """
        seed_results = []

        # Run experiment with each random seed
        for seed in self.config.random_forest.random_seeds:
            seed_result = self.run_single_experiment(X, y, model_params, experiment_id, seed)
            seed_results.append(seed_result)

        # Aggregate results across seeds
        numeric_metrics = ['training_time', 'prediction_time', 'training_memory_mb',
                          'prediction_memory_mb', 'training_cpu_percent', 'prediction_cpu_percent']

        # Add performance metrics based on task type
        if self.config.data.task_type == "classification":
            numeric_metrics.extend(['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'])
        else:
            numeric_metrics.extend(['mse', 'rmse'])

        # Calculate mean and std for each metric
        aggregated_result = {
            'experiment_id': experiment_id,
            'n_seeds': len(self.config.random_forest.random_seeds),
            'random_seeds': self.config.random_forest.random_seeds,
            'timestamp': time.time(),
            'n_samples_train': seed_results[0]['n_samples_train'],
            'n_samples_test': seed_results[0]['n_samples_test'],
            'n_features': seed_results[0]['n_features'],
            'task_type': seed_results[0]['task_type'],
            'model_params': seed_results[0]['model_params'],
        }

        # Aggregate numeric metrics
        for metric in numeric_metrics:
            values = [result[metric] for result in seed_results if metric in result]
            if values:
                aggregated_result[f'{metric}_mean'] = np.mean(values)
                aggregated_result[f'{metric}_std'] = np.std(values)
                aggregated_result[f'{metric}_min'] = np.min(values)
                aggregated_result[f'{metric}_max'] = np.max(values)

        return aggregated_result
    
    def run_data_scaling_experiment(self) -> List[Dict[str, Any]]:
        """
        Run experiments with different dataset sizes.

        Returns:
            List of experiment results
        """
        results = []

        # Get datasets based on configuration
        if self.config.data.dataset_name == "adult":
            # Use Adult dataset with specified sample sizes
            datasets = self.adult_loader.get_dataset(sample_sizes=self.config.data.sample_sizes)
        else:
            # Generate synthetic datasets with different scales
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
            'n_jobs': self.config.random_forest.n_jobs
        }

        for dataset_name, (X, y) in datasets.items():
            experiment_id = f"data_scaling_{dataset_name}"

            # Run with multiple seeds and aggregate results
            result = self.run_multi_seed_experiment(X, y, default_params, experiment_id)
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

        # Get dataset based on configuration
        if self.config.data.dataset_name == "adult":
            # Use a subset of the Adult dataset for parameter scaling
            datasets = self.adult_loader.get_dataset(sample_sizes=[self.config.data.base_samples])
            X, y = list(datasets.values())[0]
        else:
            # Generate a single synthetic dataset
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

            # Run with multiple seeds and aggregate results
            result = self.run_multi_seed_experiment(X, y, params, experiment_id)
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