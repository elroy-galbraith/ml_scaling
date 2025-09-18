"""
Data loading and preprocessing utilities for Random Forest scaling research.

This module provides functions to load datasets, create train/test splits,
and generate subsamples of specific sizes for scaling experiments.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Union, Optional, List
import warnings

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config

# Set up logging
logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class DataLoader:
    """Class for loading and preprocessing datasets for scaling experiments."""
    
    def __init__(self, dataset_name: str = "adult"):
        """
        Initialize the DataLoader.
        
        Args:
            dataset_name: Name of the dataset to load. Supports 'adult' or 'synthetic'.
        """
        self.dataset_name = dataset_name
        self.label_encoder = LabelEncoder()
        self._X_full = None
        self._y_full = None
        self._feature_names = None
    
    def load_synthetic_dataset(self, n_samples: int = 32561, n_features: int = 14) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create a synthetic dataset similar to the Adult dataset for testing.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features to generate
            
        Returns:
            Tuple of (features, target) as pandas DataFrame and Series.
        """
        logger.info(f"Generating synthetic dataset: {n_samples} samples, {n_features} features")
        
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            n_redundant=int(n_features * 0.2),
            n_clusters_per_class=2,
            flip_y=0.01,
            class_sep=0.8,
            random_state=42
        )
        
        # Convert to DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")
        
        # Add some categorical-like features by binning
        for i in range(0, min(4, n_features), 2):
            X_df[f"feature_{i}"] = pd.cut(X_df[f"feature_{i}"], bins=5, labels=False)
        
        self._X_full = X_df
        self._y_full = y_series
        self._feature_names = feature_names
        
        logger.info(f"Synthetic dataset created: {X_df.shape[0]} samples, {X_df.shape[1]} features")
        return X_df, y_series
    
    def load_adult_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the Adult/Census dataset from OpenML.
        
        Returns:
            Tuple of (features, target) as pandas DataFrame and Series.
        """
        logger.info("Loading Adult/Census dataset from OpenML...")
        
        try:
            # Load the adult dataset (id=1590 is the adult dataset)
            adult = fetch_openml(name='adult', version=2, as_frame=True, parser='auto')
            X, y = adult.data, adult.target
            
            logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Handle categorical variables
            categorical_features = X.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_features) > 0:
                logger.info(f"Encoding {len(categorical_features)} categorical features")
                X_encoded = X.copy()
                
                for col in categorical_features:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X[col].astype(str))
            else:
                X_encoded = X
            
            # Encode target variable
            if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                y_encoded = pd.Series(self.label_encoder.fit_transform(y))
                logger.info(f"Target encoded: {dict(enumerate(self.label_encoder.classes_))}")
            else:
                y_encoded = y
            
            self._X_full = X_encoded
            self._y_full = y_encoded
            self._feature_names = X.columns.tolist()
            
            return X_encoded, y_encoded
            
        except Exception as e:
            logger.warning(f"Failed to load Adult dataset from OpenML: {e}")
            logger.info("Falling back to synthetic dataset")
            return self.load_synthetic_dataset()
    
    def load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the specified dataset.
        
        Returns:
            Tuple of (features, target) as pandas DataFrame and Series.
        """
        if self.dataset_name.lower() == "adult":
            return self.load_adult_dataset()
        elif self.dataset_name.lower() == "synthetic":
            return self.load_synthetic_dataset()
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' not supported. "
                           f"Currently supported: 'adult', 'synthetic'")
    
    def create_train_test_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = None,
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create stratified train/test split.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = 1 - config.TRAIN_TEST_SPLIT
        
        if random_state is None:
            random_state = config.RANDOM_SEEDS[0]
        
        logger.info(f"Creating train/test split with test_size={test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if config.STRATIFY else None
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def create_subsample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        size: Union[int, str],
        random_state: int = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create a subsample of the dataset.
        
        Args:
            X: Feature matrix
            y: Target vector
            size: Number of samples or 'full' for complete dataset
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_subsample, y_subsample)
        """
        if random_state is None:
            random_state = config.RANDOM_SEEDS[0]
        
        if size == "full" or size >= len(X):
            logger.info("Using full dataset")
            return X, y
        
        logger.info(f"Creating subsample of size {size}")
        
        # Stratified sampling to maintain class distribution
        X_sub, _, y_sub, _ = train_test_split(
            X, y,
            train_size=size,
            random_state=random_state,
            stratify=y if config.STRATIFY and size < len(y) else None
        )
        
        logger.info(f"Subsample created: {X_sub.shape[0]} samples")
        
        return X_sub, y_sub
    
    def get_data_sizes_samples(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sizes: List[Union[int, str]] = None
    ) -> List[Tuple[pd.DataFrame, pd.Series, Union[int, str]]]:
        """
        Generate multiple subsamples for data size scaling experiments.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            sizes: List of sample sizes to generate
            
        Returns:
            List of tuples (X_subsample, y_subsample, size)
        """
        if sizes is None:
            sizes = config.DATA_SIZES
        
        samples = []
        for size in sizes:
            if size == "full" or size >= len(X_train):
                X_sub, y_sub = X_train, y_train
                actual_size = len(X_train)
            else:
                X_sub, y_sub = self.create_subsample(
                    X_train, y_train, size, config.RANDOM_SEEDS[0]
                )
                actual_size = size
            
            samples.append((X_sub, y_sub, actual_size))
        
        return samples
    
    @property
    def feature_names(self) -> List[str]:
        """Get the feature names of the loaded dataset."""
        return self._feature_names
    
    @property
    def dataset_info(self) -> dict:
        """Get information about the loaded dataset."""
        if self._X_full is None:
            return {}
        
        return {
            "dataset_name": self.dataset_name,
            "n_samples": len(self._X_full),
            "n_features": self._X_full.shape[1],
            "feature_names": self._feature_names,
            "target_classes": self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None
        }


def load_and_prepare_data(dataset_name: str = "adult") -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, DataLoader
]:
    """
    Convenience function to load dataset and create train/test split.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, data_loader)
    """
    logger.info(f"Loading and preparing dataset: {dataset_name}")
    
    # Create data loader and load dataset
    data_loader = DataLoader(dataset_name)
    X, y = data_loader.load_dataset()
    
    # Create train/test split
    X_train, X_test, y_train, y_test = data_loader.create_train_test_split(X, y)
    
    logger.info("Data preparation completed successfully")
    
    return X_train, X_test, y_train, y_test, data_loader