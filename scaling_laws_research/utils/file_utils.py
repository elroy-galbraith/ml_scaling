"""
File I/O utilities for the Random Forest scaling research.

This module provides utilities for saving and loading results, handling
different file formats, and managing file paths.
"""

import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import config

logger = logging.getLogger(__name__)


class ResultsIO:
    """Class for handling I/O operations for experiment results."""
    
    @staticmethod
    def save_results(
        results: List[Dict[str, Any]], 
        filepath: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Save results to file in specified format.
        
        Args:
            results: List of experiment results
            filepath: Path to save file
            format: File format ('json', 'csv', 'pickle')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format.lower() == "csv":
            df = pd.DataFrame(results)
            df.to_csv(filepath, index=False)
        
        elif format.lower() == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {filepath} ({format} format)")
    
    @staticmethod
    def load_results(
        filepath: Union[str, Path],
        format: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        Load results from file.
        
        Args:
            filepath: Path to load file
            format: File format ('json', 'csv', 'pickle', 'auto')
            
        Returns:
            List of experiment results
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Auto-detect format from extension
        if format == "auto":
            ext = filepath.suffix.lower()
            if ext == ".json":
                format = "json"
            elif ext == ".csv":
                format = "csv"
            elif ext == ".pkl" or ext == ".pickle":
                format = "pickle"
            else:
                raise ValueError(f"Cannot auto-detect format for {filepath}")
        
        if format.lower() == "json":
            with open(filepath, 'r') as f:
                results = json.load(f)
        
        elif format.lower() == "csv":
            df = pd.read_csv(filepath)
            results = df.to_dict('records')
        
        elif format.lower() == "pickle":
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Loaded {len(results)} results from {filepath}")
        return results
    
    @staticmethod
    def save_metadata(
        metadata: Dict[str, Any],
        filepath: Union[str, Path]
    ) -> None:
        """
        Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to {filepath}")
    
    @staticmethod
    def load_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load metadata from JSON file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Metadata dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Metadata loaded from {filepath}")
        return metadata


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp_string() -> str:
    """
    Get current timestamp as a string suitable for filenames.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_latest_results(
    dataset: str,
    experiment_type: str,
    results_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Find the most recent results file for a given dataset and experiment type.
    
    Args:
        dataset: Dataset name
        experiment_type: Experiment type
        results_dir: Results directory (if None, uses config default)
        
    Returns:
        Path to most recent results file, or None if not found
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    
    search_path = results_dir / dataset / experiment_type
    
    if not search_path.exists():
        return None
    
    # Look for JSON files (preferred) or CSV files
    json_files = list(search_path.glob(f"{experiment_type}_results*.json"))
    csv_files = list(search_path.glob(f"{experiment_type}_results*.csv"))
    
    all_files = json_files + csv_files
    
    if not all_files:
        return None
    
    # Return most recent file
    latest_file = max(all_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest results: {latest_file}")
    return latest_file


def cleanup_old_results(
    dataset: str,
    experiment_type: str,
    keep_count: int = 5,
    results_dir: Optional[Path] = None
) -> None:
    """
    Clean up old result files, keeping only the most recent ones.
    
    Args:
        dataset: Dataset name
        experiment_type: Experiment type
        keep_count: Number of recent files to keep
        results_dir: Results directory (if None, uses config default)
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    
    search_path = results_dir / dataset / experiment_type
    
    if not search_path.exists():
        return
    
    # Find all result files
    result_files = list(search_path.glob(f"{experiment_type}_results*"))
    
    if len(result_files) <= keep_count:
        return
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Remove old files
    files_to_remove = result_files[keep_count:]
    for file_path in files_to_remove:
        file_path.unlink()
        logger.info(f"Removed old result file: {file_path}")
    
    logger.info(f"Cleaned up {len(files_to_remove)} old result files")


def export_results_summary(
    all_results: Dict[str, List[Dict[str, Any]]],
    analysis_results: Dict[str, Dict[str, Any]],
    output_path: Union[str, Path],
    dataset_name: str = "adult"
) -> None:
    """
    Export a summary of all results to a comprehensive report.
    
    Args:
        all_results: Dictionary of experiment results
        analysis_results: Dictionary of analysis results
        output_path: Path for output file
        dataset_name: Name of the dataset
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "dataset": dataset_name,
        "timestamp": get_timestamp_string(),
        "experiments": {},
        "summary_statistics": {}
    }
    
    # Add experiment results
    for exp_type, results in all_results.items():
        exp_summary = {
            "n_experiments": len(results),
            "experiment_type": exp_type,
            "results": results
        }
        
        # Add analysis results if available
        if exp_type in analysis_results:
            exp_summary["analysis"] = analysis_results[exp_type]
        
        summary["experiments"][exp_type] = exp_summary
    
    # Add overall summary statistics
    total_experiments = sum(len(results) for results in all_results.values())
    summary["summary_statistics"] = {
        "total_experiments": total_experiments,
        "experiment_types": list(all_results.keys()),
        "config_used": {
            "random_seeds": config.RANDOM_SEEDS,
            "n_repetitions": config.N_REPETITIONS,
            "data_sizes": config.DATA_SIZES,
            "tree_counts": config.TREE_COUNTS,
            "max_depths": config.MAX_DEPTHS
        }
    }
    
    # Save summary
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Results summary exported to {output_path}")