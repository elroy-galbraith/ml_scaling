"""Adult/Census Income dataset loader and preprocessor for Random Forest scaling experiments."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdultDatasetLoader:
    """
    Loads and preprocesses the Adult/Census Income dataset from UCI ML Repository.

    The Adult dataset contains 48,842 instances with 14 attributes, including
    both categorical and continuous features. The prediction task is to determine
    whether a person makes over 50K a year.
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the Adult dataset loader.

        Args:
            data_dir: Directory to store raw data files
        """
        self.data_dir = data_dir
        self.train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        self.test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

        # Column names as per UCI repository documentation
        self.column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
        ]

        # Categorical columns that need encoding
        self.categorical_columns = [
            'workclass', 'education', 'marital_status', 'occupation',
            'relationship', 'race', 'sex', 'native_country'
        ]

        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

        self.label_encoders = {}
        self.scaler = StandardScaler()
        self._data_loaded = False
        self._preprocessed = False

    def download_data(self) -> None:
        """Download the Adult dataset from UCI ML Repository."""
        train_path = os.path.join(self.data_dir, "adult.data")
        test_path = os.path.join(self.data_dir, "adult.test")

        if not os.path.exists(train_path):
            logger.info("Downloading training data...")
            urllib.request.urlretrieve(self.train_url, train_path)
            logger.info(f"Training data saved to {train_path}")

        if not os.path.exists(test_path):
            logger.info("Downloading test data...")
            urllib.request.urlretrieve(self.test_url, test_path)
            logger.info(f"Test data saved to {test_path}")

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw data from downloaded files.

        Returns:
            Tuple of (train_df, test_df)
        """
        self.download_data()

        train_path = os.path.join(self.data_dir, "adult.data")
        test_path = os.path.join(self.data_dir, "adult.test")

        # Load training data
        train_df = pd.read_csv(
            train_path,
            names=self.column_names,
            skipinitialspace=True,
            na_values=" ?"
        )

        # Load test data (skip first line which contains header info)
        test_df = pd.read_csv(
            test_path,
            names=self.column_names,
            skipinitialspace=True,
            skiprows=1,
            na_values=" ?"
        )

        # Clean the income column in test data (remove periods)
        test_df['income'] = test_df['income'].str.replace('.', '', regex=False)

        logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")

        self._data_loaded = True
        return train_df, test_df

    def preprocess_data(self, df: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
        """
        Preprocess the Adult dataset.

        Args:
            df: Raw dataframe
            fit_encoders: Whether to fit label encoders (True for train, False for test)

        Returns:
            Preprocessed dataframe
        """
        df = df.copy()

        # Handle missing values
        for col in self.categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')

        # Fill numerical missing values with median
        numerical_columns = ['age', 'fnlwgt', 'education_num', 'capital_gain',
                           'capital_loss', 'hours_per_week']
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].median())

        # Encode categorical variables
        for col in self.categorical_columns:
            if fit_encoders:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories in test data
                    df[col] = df[col].astype(str)
                    mask = df[col].isin(self.label_encoders[col].classes_)
                    df.loc[~mask, col] = self.label_encoders[col].classes_[0]  # Use most frequent
                    df[col] = self.label_encoders[col].transform(df[col])

        # Encode target variable
        target_col = 'income'
        if fit_encoders:
            if target_col not in self.label_encoders:
                self.label_encoders[target_col] = LabelEncoder()
            df[target_col] = self.label_encoders[target_col].fit_transform(df[target_col].astype(str))
        else:
            if target_col in self.label_encoders:
                df[target_col] = df[target_col].astype(str)
                mask = df[target_col].isin(self.label_encoders[target_col].classes_)
                df.loc[~mask, target_col] = self.label_encoders[target_col].classes_[0]
                df[target_col] = self.label_encoders[target_col].transform(df[target_col])

        logger.info(f"Preprocessed dataframe shape: {df.shape}")
        return df

    def get_dataset(self, sample_sizes: List[int] = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get preprocessed dataset with different sample sizes.

        Args:
            sample_sizes: List of sample sizes to create. If None, uses research plan defaults.

        Returns:
            Dictionary mapping sample size names to (X, y) tuples
        """
        if sample_sizes is None:
            sample_sizes = [500, 1000, 2000, 5000, 10000, 20000, 38000]

        # Load and preprocess data
        train_df, test_df = self.load_raw_data()

        # Combine train and test for consistent preprocessing
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        processed_df = self.preprocess_data(combined_df, fit_encoders=True)

        # Separate features and target
        X = processed_df.drop('income', axis=1).values
        y = processed_df['income'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        datasets = {}

        for sample_size in sample_sizes:
            if sample_size > len(X_scaled):
                logger.warning(f"Requested sample size {sample_size} exceeds dataset size {len(X_scaled)}")
                sample_size = len(X_scaled)

            # Stratified sampling to maintain class balance
            X_sample, _, y_sample, _ = train_test_split(
                X_scaled, y,
                train_size=sample_size,
                stratify=y,
                random_state=42
            )

            datasets[f"adult_{sample_size}"] = (X_sample, y_sample)
            logger.info(f"Created dataset with {sample_size} samples, "
                       f"class distribution: {np.bincount(y_sample)}")

        self._preprocessed = True
        return datasets

    def get_full_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the complete preprocessed Adult dataset.

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        datasets = self.get_dataset(sample_sizes=[38000])  # Use full available data
        return list(datasets.values())[0]

    def get_dataset_info(self) -> Dict[str, any]:
        """
        Get information about the Adult dataset.

        Returns:
            Dictionary containing dataset information
        """
        if not self._data_loaded:
            train_df, test_df = self.load_raw_data()
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
        else:
            # Reload for info
            train_df, test_df = self.load_raw_data()
            combined_df = pd.concat([train_df, test_df], ignore_index=True)

        return {
            'name': 'Adult/Census Income Dataset',
            'total_samples': len(combined_df),
            'n_features': len(self.column_names) - 1,  # Exclude target
            'n_categorical_features': len(self.categorical_columns),
            'n_numerical_features': len(self.column_names) - 1 - len(self.categorical_columns),
            'target_classes': combined_df['income'].value_counts().to_dict(),
            'missing_values': combined_df.isnull().sum().sum(),
            'categorical_columns': self.categorical_columns,
            'column_names': self.column_names[:-1]  # Exclude target
        }


def load_adult_dataset(sample_sizes: List[int] = None, data_dir: str = "data/raw") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function to load Adult dataset with specified sample sizes.

    Args:
        sample_sizes: List of sample sizes to create
        data_dir: Directory to store raw data files

    Returns:
        Dictionary mapping sample size names to (X, y) tuples
    """
    loader = AdultDatasetLoader(data_dir=data_dir)
    return loader.get_dataset(sample_sizes=sample_sizes)


if __name__ == "__main__":
    # Test the Adult dataset loader
    print("Testing Adult Dataset Loader...")

    loader = AdultDatasetLoader()
    info = loader.get_dataset_info()

    print("Dataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test with small sample sizes
    test_sample_sizes = [500, 1000, 2000]
    datasets = loader.get_dataset(sample_sizes=test_sample_sizes)

    print(f"\nLoaded {len(datasets)} datasets:")
    for name, (X, y) in datasets.items():
        print(f"  {name}: X shape={X.shape}, y shape={y.shape}, "
              f"class balance={np.bincount(y)}")