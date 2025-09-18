"""Data loading utilities for benchmark datasets."""

import numpy as np
from sklearn.datasets import (
    load_breast_cancer, load_wine, load_digits,
    fetch_20newsgroups_vectorized, fetch_covtype
)
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
import warnings


def load_benchmark_datasets(
    test_size: float = 0.2, 
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Load benchmark datasets for scaling experiments.
    
    Args:
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducible splits
        
    Returns:
        Dictionary mapping dataset names to dataset information
    """
    datasets = {}
    
    # Small datasets
    try:
        # Breast cancer dataset
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, test_size=test_size, random_state=random_state
        )
        datasets['breast_cancer'] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'task_type': 'classification',
            'n_samples': len(cancer.data),
            'n_features': cancer.data.shape[1],
            'n_classes': len(np.unique(cancer.target)),
            'description': 'Breast cancer classification dataset'
        }
    except Exception as e:
        warnings.warn(f"Could not load breast cancer dataset: {e}")
    
    try:
        # Wine dataset
        wine = load_wine()
        X_train, X_test, y_train, y_test = train_test_split(
            wine.data, wine.target, test_size=test_size, random_state=random_state
        )
        datasets['wine'] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'task_type': 'classification',
            'n_samples': len(wine.data),
            'n_features': wine.data.shape[1],
            'n_classes': len(np.unique(wine.target)),
            'description': 'Wine classification dataset'
        }
    except Exception as e:
        warnings.warn(f"Could not load wine dataset: {e}")
    
    try:
        # Digits dataset (scaled down for efficiency)
        digits = load_digits()
        # Take a subset for faster experiments
        subset_size = min(1000, len(digits.data))
        indices = np.random.RandomState(random_state).choice(
            len(digits.data), subset_size, replace=False
        )
        X_subset = digits.data[indices]
        y_subset = digits.target[indices]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_subset, test_size=test_size, random_state=random_state
        )
        datasets['digits'] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'task_type': 'classification',
            'n_samples': len(X_subset),
            'n_features': X_subset.shape[1],
            'n_classes': len(np.unique(y_subset)),
            'description': 'Handwritten digits classification (subset)'
        }
    except Exception as e:
        warnings.warn(f"Could not load digits dataset: {e}")
    
    # Large datasets (optional, may take time to download)
    try:
        # 20 Newsgroups (vectorized version)
        newsgroups = fetch_20newsgroups_vectorized(subset='train', random_state=random_state)
        # Take a subset for efficiency
        subset_size = min(2000, len(newsgroups.data))
        indices = np.random.RandomState(random_state).choice(
            len(newsgroups.data), subset_size, replace=False
        )
        X_subset = newsgroups.data[indices].toarray()
        y_subset = newsgroups.target[indices]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_subset, test_size=test_size, random_state=random_state
        )
        datasets['newsgroups'] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'task_type': 'classification',
            'n_samples': len(X_subset),
            'n_features': X_subset.shape[1],
            'n_classes': len(np.unique(y_subset)),
            'description': '20 Newsgroups text classification (subset)'
        }
    except Exception as e:
        warnings.warn(f"Could not load 20 newsgroups dataset: {e}")
    
    return datasets


def get_dataset_info(datasets: Dict[str, Dict[str, Any]]) -> None:
    """
    Print information about loaded datasets.
    
    Args:
        datasets: Dictionary of datasets from load_benchmark_datasets
    """
    print("Available Benchmark Datasets:")
    print("=" * 50)
    
    for name, data in datasets.items():
        print(f"\n{name.title()}:")
        print(f"  Task: {data['task_type']}")
        print(f"  Samples: {data['n_samples']}")
        print(f"  Features: {data['n_features']}")
        if data['task_type'] == 'classification':
            print(f"  Classes: {data['n_classes']}")
        print(f"  Train/Test split: {len(data['X_train'])}/{len(data['X_test'])}")
        print(f"  Description: {data['description']}")


def load_single_dataset(
    name: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single benchmark dataset by name.
    
    Args:
        name: Dataset name ('breast_cancer', 'wine', 'digits', 'newsgroups')
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducible splits
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    datasets = load_benchmark_datasets(test_size=test_size, random_state=random_state)
    
    if name not in datasets:
        available = list(datasets.keys())
        raise ValueError(f"Dataset '{name}' not available. Available datasets: {available}")
    
    data = datasets[name]
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']