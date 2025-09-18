"""
Scaling experiments module for Random Forest research.

This module contains classes and functions to run systematic scaling experiments
on Random Forest models to discover power-law relationships.
"""

import logging
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config
from scaling_laws_research.data import DataLoader, load_and_prepare_data

# Set up logging
logger = logging.getLogger(__name__)


class ScalingExperiments:
    """Class for running systematic scaling experiments on Random Forest models."""
    
    def __init__(self, dataset_name: str = "adult"):
        """
        Initialize the scaling experiments.
        
        Args:
            dataset_name: Name of the dataset to use for experiments.
        """
        self.dataset_name = dataset_name
        self.data_loader = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = []
        
        # Create directories
        config.create_directories()
    
    def load_data(self) -> None:
        """Load and prepare the dataset for experiments."""
        logger.info(f"Loading dataset: {self.dataset_name}")
        self.X_train, self.X_test, self.y_train, self.y_test, self.data_loader = \
            load_and_prepare_data(self.dataset_name)
        logger.info("Dataset loaded successfully")
    
    def _evaluate_model(
        self, 
        model: RandomForestClassifier, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate a trained Random Forest model.
        
        Args:
            model: Trained Random Forest model
            X_train: Training features
            X_test: Test features  
            y_train: Training targets
            y_test: Test targets
            
        Returns:
            Dictionary of performance metrics
        """
        # Training time is already recorded during fitting
        
        # Measure prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "prediction_time": prediction_time
        }
        
        return metrics
    
    def _run_single_experiment(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        rf_params: Dict[str, Any],
        experiment_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single experiment with given parameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            rf_params: Random Forest parameters
            experiment_info: Information about the experiment
            
        Returns:
            Dictionary containing experiment results
        """
        # Create and train the model
        model = RandomForestClassifier(**rf_params)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate the model
        metrics = self._evaluate_model(model, X_train, self.X_test, y_train, self.y_test)
        metrics["training_time"] = training_time
        
        # Combine results
        result = {
            **experiment_info,
            **metrics,
            "model_params": rf_params,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def data_size_scaling(
        self, 
        sizes: List[Union[int, str]] = None,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run data size scaling experiments.
        
        Args:
            sizes: List of data sizes to test
            save_results: Whether to save results to file
            
        Returns:
            List of experiment results
        """
        if self.X_train is None:
            self.load_data()
        
        if sizes is None:
            sizes = config.DATA_SIZES
        
        logger.info("Starting data size scaling experiments")
        results = []
        
        # Get subsamples for different sizes
        samples = self.data_loader.get_data_sizes_samples(self.X_train, self.y_train, sizes)
        
        # Run experiments for each size and each repetition
        total_experiments = len(samples) * config.N_REPETITIONS
        progress_bar = tqdm(total=total_experiments, desc="Data size scaling")
        
        for X_sub, y_sub, actual_size in samples:
            for rep in range(config.N_REPETITIONS):
                # Prepare RF parameters
                rf_params = config.RF_DEFAULT_PARAMS.copy()
                rf_params["random_state"] = config.RANDOM_SEEDS[rep]
                
                # Experiment info
                experiment_info = {
                    "experiment_type": "data_size_scaling",
                    "dataset": self.dataset_name,
                    "data_size": actual_size,
                    "repetition": rep + 1,
                    "n_estimators": rf_params["n_estimators"],
                    "max_depth": rf_params["max_depth"]
                }
                
                # Run experiment
                try:
                    result = self._run_single_experiment(X_sub, y_sub, rf_params, experiment_info)
                    results.append(result)
                    logger.debug(f"Completed data size {actual_size}, rep {rep + 1}")
                except Exception as e:
                    logger.error(f"Failed experiment for size {actual_size}, rep {rep + 1}: {e}")
                
                progress_bar.update(1)
        
        progress_bar.close()
        
        if save_results:
            self._save_results(results, "data_size_scaling")
        
        logger.info(f"Completed {len(results)} data size scaling experiments")
        return results
    
    def tree_count_scaling(
        self,
        tree_counts: List[int] = None,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run tree count scaling experiments.
        
        Args:
            tree_counts: List of tree counts to test
            save_results: Whether to save results to file
            
        Returns:
            List of experiment results
        """
        if self.X_train is None:
            self.load_data()
        
        if tree_counts is None:
            tree_counts = config.TREE_COUNTS
        
        logger.info("Starting tree count scaling experiments")
        results = []
        
        # Run experiments for each tree count and each repetition
        total_experiments = len(tree_counts) * config.N_REPETITIONS
        progress_bar = tqdm(total=total_experiments, desc="Tree count scaling")
        
        for n_trees in tree_counts:
            for rep in range(config.N_REPETITIONS):
                # Prepare RF parameters
                rf_params = config.RF_DEFAULT_PARAMS.copy()
                rf_params["n_estimators"] = n_trees
                rf_params["random_state"] = config.RANDOM_SEEDS[rep]
                
                # Experiment info
                experiment_info = {
                    "experiment_type": "tree_count_scaling",
                    "dataset": self.dataset_name,
                    "data_size": len(self.X_train),
                    "repetition": rep + 1,
                    "n_estimators": n_trees,
                    "max_depth": rf_params["max_depth"]
                }
                
                # Run experiment
                try:
                    result = self._run_single_experiment(
                        self.X_train, self.y_train, rf_params, experiment_info
                    )
                    results.append(result)
                    logger.debug(f"Completed tree count {n_trees}, rep {rep + 1}")
                except Exception as e:
                    logger.error(f"Failed experiment for tree count {n_trees}, rep {rep + 1}: {e}")
                
                progress_bar.update(1)
        
        progress_bar.close()
        
        if save_results:
            self._save_results(results, "tree_count_scaling")
        
        logger.info(f"Completed {len(results)} tree count scaling experiments")
        return results
    
    def depth_scaling(
        self,
        max_depths: List[Optional[int]] = None,
        save_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run max depth scaling experiments.
        
        Args:
            max_depths: List of max depths to test
            save_results: Whether to save results to file
            
        Returns:
            List of experiment results
        """
        if self.X_train is None:
            self.load_data()
        
        if max_depths is None:
            max_depths = config.MAX_DEPTHS
        
        logger.info("Starting depth scaling experiments")
        results = []
        
        # Run experiments for each max depth and each repetition
        total_experiments = len(max_depths) * config.N_REPETITIONS
        progress_bar = tqdm(total=total_experiments, desc="Depth scaling")
        
        for max_depth in max_depths:
            for rep in range(config.N_REPETITIONS):
                # Prepare RF parameters
                rf_params = config.RF_DEFAULT_PARAMS.copy()
                rf_params["max_depth"] = max_depth
                rf_params["random_state"] = config.RANDOM_SEEDS[rep]
                
                # Experiment info
                experiment_info = {
                    "experiment_type": "depth_scaling",
                    "dataset": self.dataset_name,
                    "data_size": len(self.X_train),
                    "repetition": rep + 1,
                    "n_estimators": rf_params["n_estimators"],
                    "max_depth": max_depth
                }
                
                # Run experiment
                try:
                    result = self._run_single_experiment(
                        self.X_train, self.y_train, rf_params, experiment_info
                    )
                    results.append(result)
                    logger.debug(f"Completed max depth {max_depth}, rep {rep + 1}")
                except Exception as e:
                    logger.error(f"Failed experiment for max depth {max_depth}, rep {rep + 1}: {e}")
                
                progress_bar.update(1)
        
        progress_bar.close()
        
        if save_results:
            self._save_results(results, "depth_scaling")
        
        logger.info(f"Completed {len(results)} depth scaling experiments")
        return results
    
    def run_all_experiments(self, save_results: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run all scaling experiments.
        
        Args:
            save_results: Whether to save results to files
            
        Returns:
            Dictionary containing all experiment results
        """
        logger.info("Starting all scaling experiments")
        
        all_results = {
            "data_size_scaling": self.data_size_scaling(save_results=save_results),
            "tree_count_scaling": self.tree_count_scaling(save_results=save_results),
            "depth_scaling": self.depth_scaling(save_results=save_results)
        }
        
        logger.info("All scaling experiments completed")
        return all_results
    
    def _save_results(self, results: List[Dict[str, Any]], experiment_type: str) -> None:
        """
        Save experiment results to files.
        
        Args:
            results: List of experiment results
            experiment_type: Type of experiment
        """
        results_path = config.get_results_path(self.dataset_name, experiment_type)
        
        # Save as CSV
        df = pd.DataFrame(results)
        csv_path = results_path / f"{experiment_type}_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON for complete information
        json_path = results_path / f"{experiment_type}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metadata
        metadata = {
            "experiment_type": experiment_type,
            "dataset": self.dataset_name,
            "n_experiments": len(results),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "random_seeds": config.RANDOM_SEEDS,
                "n_repetitions": config.N_REPETITIONS,
                "rf_default_params": config.RF_DEFAULT_PARAMS
            }
        }
        
        if self.data_loader:
            metadata["dataset_info"] = self.data_loader.dataset_info
        
        metadata_path = results_path / f"{experiment_type}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def load_results(self, experiment_type: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load previously saved experiment results.
        
        Args:
            experiment_type: Type of experiment to load
            
        Returns:
            List of experiment results or None if not found
        """
        results_path = config.get_results_path(self.dataset_name, experiment_type)
        json_path = results_path / f"{experiment_type}_results.json"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} results from {json_path}")
            return results
        else:
            logger.warning(f"No results found at {json_path}")
            return None