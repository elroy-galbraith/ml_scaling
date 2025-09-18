"""
Main pipeline for Random Forest scaling laws research.

This script orchestrates the full experimental pipeline with a command-line
interface for running specific experiments or the complete pipeline.
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from scaling_laws_research.experiments import ScalingExperiments
from scaling_laws_research.analysis import analyze_scaling_results
from scaling_laws_research.visualizations import create_all_visualizations
from scaling_laws_research.utils import setup_logging, get_logger, ResultsIO
from scaling_laws_research.utils.file_utils import export_results_summary


def run_data_scaling_experiment(
    dataset: str = "adult",
    save_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Run data size scaling experiment.
    
    Args:
        dataset: Dataset name
        save_results: Whether to save results
        
    Returns:
        List of experiment results
    """
    logger = get_logger("main")
    logger.info("Starting data size scaling experiment")
    
    experiments = ScalingExperiments(dataset)
    results = experiments.data_size_scaling(save_results=save_results)
    
    logger.info(f"Data size scaling experiment completed: {len(results)} experiments")
    return results


def run_tree_scaling_experiment(
    dataset: str = "adult",
    save_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Run tree count scaling experiment.
    
    Args:
        dataset: Dataset name
        save_results: Whether to save results
        
    Returns:
        List of experiment results
    """
    logger = get_logger("main")
    logger.info("Starting tree count scaling experiment")
    
    experiments = ScalingExperiments(dataset)
    results = experiments.tree_count_scaling(save_results=save_results)
    
    logger.info(f"Tree count scaling experiment completed: {len(results)} experiments")
    return results


def run_depth_scaling_experiment(
    dataset: str = "adult", 
    save_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Run depth scaling experiment.
    
    Args:
        dataset: Dataset name
        save_results: Whether to save results
        
    Returns:
        List of experiment results
    """
    logger = get_logger("main")
    logger.info("Starting depth scaling experiment")
    
    experiments = ScalingExperiments(dataset)
    results = experiments.depth_scaling(save_results=save_results)
    
    logger.info(f"Depth scaling experiment completed: {len(results)} experiments")
    return results


def run_all_experiments(
    dataset: str = "adult",
    save_results: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run all scaling experiments.
    
    Args:
        dataset: Dataset name
        save_results: Whether to save results
        
    Returns:
        Dictionary of all experiment results
    """
    logger = get_logger("main")
    logger.info("Starting all scaling experiments")
    
    experiments = ScalingExperiments(dataset)
    all_results = experiments.run_all_experiments(save_results=save_results)
    
    total_experiments = sum(len(results) for results in all_results.values())
    logger.info(f"All scaling experiments completed: {total_experiments} total experiments")
    
    return all_results


def analyze_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    dataset: str = "adult",
    save_analysis: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze scaling experiment results.
    
    Args:
        all_results: Dictionary of experiment results
        dataset: Dataset name
        save_analysis: Whether to save analysis results
        
    Returns:
        Dictionary of analysis results
    """
    logger = get_logger("main")
    logger.info("Starting analysis of scaling results")
    
    analysis_results = {}
    
    for exp_type, results in all_results.items():
        logger.info(f"Analyzing {exp_type}")
        analysis = analyze_scaling_results(results, exp_type)
        analysis_results[exp_type] = analysis
        
        if save_analysis:
            # Save analysis results
            analysis_path = config.get_results_path(dataset, exp_type)
            ResultsIO.save_results(
                [analysis], 
                analysis_path / f"{exp_type}_analysis.json"
            )
    
    logger.info("Analysis completed")
    return analysis_results


def generate_visualizations(
    all_results: Dict[str, List[Dict[str, Any]]],
    analysis_results: Dict[str, Dict[str, Any]],
    dataset: str = "adult"
) -> Dict[str, Any]:
    """
    Generate all visualizations.
    
    Args:
        all_results: Dictionary of experiment results
        analysis_results: Dictionary of analysis results
        dataset: Dataset name
        
    Returns:
        Dictionary of generated figures
    """
    logger = get_logger("main")
    logger.info("Generating visualizations")
    
    figures = create_all_visualizations(all_results, analysis_results, dataset)
    
    logger.info(f"Generated {len(figures)} visualizations")
    return figures


def load_existing_results(
    dataset: str,
    experiment_types: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load existing experiment results.
    
    Args:
        dataset: Dataset name
        experiment_types: List of experiment types to load (if None, load all)
        
    Returns:
        Dictionary of loaded results
    """
    logger = get_logger("main")
    
    if experiment_types is None:
        experiment_types = ["data_size_scaling", "tree_count_scaling", "depth_scaling"]
    
    all_results = {}
    
    for exp_type in experiment_types:
        experiments = ScalingExperiments(dataset)
        results = experiments.load_results(exp_type)
        
        if results:
            all_results[exp_type] = results
            logger.info(f"Loaded {len(results)} results for {exp_type}")
        else:
            logger.warning(f"No existing results found for {exp_type}")
    
    return all_results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Random Forest Scaling Laws Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run data scaling experiment
  python main.py --experiment data_scaling --dataset adult
  
  # Run all experiments
  python main.py --experiment all --dataset adult
  
  # Generate visualizations from existing results
  python main.py --visualize --dataset adult
  
  # Run analysis only
  python main.py --analyze --dataset adult
        """
    )
    
    parser.add_argument(
        "--experiment", "-e",
        choices=["data_scaling", "tree_scaling", "depth_scaling", "all"],
        help="Type of experiment to run"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        default="adult",
        help="Dataset to use (default: adult)"
    )
    
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Generate visualizations from existing results"
    )
    
    parser.add_argument(
        "--analyze", "-a", 
        action="store_true",
        help="Run analysis on existing results"
    )
    
    parser.add_argument(
        "--results-dir", "-r",
        type=Path,
        help="Custom results directory"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_file = config.RESULTS_DIR / "logs" / f"scaling_research_{args.dataset}.log"
    logger = setup_logging(args.log_level, log_file)
    
    # Override results directory if specified
    if args.results_dir:
        config.RESULTS_DIR = args.results_dir
    
    # Create directories
    config.create_directories()
    
    save_results = not args.no_save
    
    try:
        logger.info(f"Starting Random Forest scaling research for {args.dataset} dataset")
        logger.info(f"Configuration: save_results={save_results}, log_level={args.log_level}")
        
        all_results = {}
        analysis_results = {}
        
        # Run experiments
        if args.experiment:
            if args.experiment == "data_scaling":
                results = run_data_scaling_experiment(args.dataset, save_results)
                all_results["data_size_scaling"] = results
                
            elif args.experiment == "tree_scaling":
                results = run_tree_scaling_experiment(args.dataset, save_results)
                all_results["tree_count_scaling"] = results
                
            elif args.experiment == "depth_scaling":
                results = run_depth_scaling_experiment(args.dataset, save_results)
                all_results["depth_scaling"] = results
                
            elif args.experiment == "all":
                all_results = run_all_experiments(args.dataset, save_results)
        
        # Load existing results if no experiments run
        if not all_results and (args.visualize or args.analyze):
            all_results = load_existing_results(args.dataset)
        
        # Run analysis
        if all_results and (args.analyze or args.experiment or args.visualize):
            analysis_results = analyze_results(all_results, args.dataset, save_results)
        
        # Generate visualizations
        if args.visualize or args.experiment:
            if all_results and analysis_results:
                figures = generate_visualizations(all_results, analysis_results, args.dataset)
                logger.info("Visualizations saved to plots directory")
            else:
                logger.warning("No results available for visualization")
        
        # Export comprehensive summary
        if all_results and save_results:
            summary_path = config.get_results_path(args.dataset, "summary") / "complete_summary.json"
            export_results_summary(all_results, analysis_results, summary_path, args.dataset)
            logger.info(f"Complete summary exported to {summary_path}")
        
        logger.info("Pipeline completed successfully!")
        
        # Print summary
        if all_results:
            print("\n" + "="*60)
            print("EXPERIMENT SUMMARY")
            print("="*60)
            for exp_type, results in all_results.items():
                print(f"{exp_type:20s}: {len(results):4d} experiments")
            
            total = sum(len(results) for results in all_results.values())
            print(f"{'TOTAL':20s}: {total:4d} experiments")
            print("="*60)
            
            if analysis_results:
                print("\nKEY FINDINGS:")
                for exp_type, analysis in analysis_results.items():
                    if "scaling_relationships" in analysis:
                        print(f"\n{exp_type.replace('_', ' ').title()}:")
                        for metric, fit_result in analysis["scaling_relationships"].items():
                            if fit_result.get("success", False):
                                r_sq = fit_result["r_squared"]
                                func = fit_result["function"]
                                print(f"  {metric:15s}: {func:15s} (R² = {r_sq:.3f})")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()