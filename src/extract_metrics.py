#!/usr/bin/env python
"""
Usage:
    # To only merge existing metrics.csv files for a single experiment:
    python extract_metrics.py --base-dir path/to/experiment_dir --merge-only

    # To only merge existing metrics.csv files for multiple experiments:
    python extract_metrics.py --base-dir path/to/container_dir --merge-only

    # To compute metrics and then merge them for a single experiment:
    python extract_metrics.py --base-dir path/to/experiment_dir
    python extract_metrics.py --base-dir path/to/experiment_dir --merge-only

    # To compute metrics and then merge them for multiple experiments:
    python extract_metrics.py --base-dir path/to/container_dir
    python extract_metrics.py --base-dir path/to/container_dir --merge-only
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Import required dependencies
import ray
import networkx as nx
import infomeasure as im
import antropy as ant

from load_utils import (
    load_experiment_summary,
    list_available_steps,
    load_step_data,
    validate_base_dir
)
from measures import (
    entropic_measures,
    lz_complexity_measures,
    sample_entropy_measures,
    global_metrics_directed
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Helper functions
def keep_indices(removed_ids: np.ndarray, n_nodes: int) -> np.ndarray:
    """Return indices to keep after removing specified nodes."""
    mask = np.ones(n_nodes, dtype=bool)
    mask[removed_ids] = False
    return np.where(mask)[0]

def take_submatrix(M: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Extract submatrix by taking rows and columns specified by idx."""
    return M[np.ix_(idx, idx)]

@ray.remote
def process_batch_ray(batch: int, base_S_hist: np.ndarray, pruned_S_hist_batch: np.ndarray,
                 W0: np.ndarray, removed_ids: np.ndarray, n_nodes: int, I_ext: Optional[np.ndarray] = None) -> Dict:
    """
    Process a single batch - modified for Ray parallelization

    Args:
        batch: Batch index
        base_S_hist: Baseline spike history (pre-attack)
        pruned_S_hist_batch: Pruned spike history for this batch
        W0: Initial weight matrix
        removed_ids: Indices of removed nodes for this batch
        n_nodes: Total number of nodes in the original network
        I_ext: External input current (optional)

    Returns:
        Dictionary containing calculated metrics
    """
    try:
        take_idx = keep_indices(removed_ids[batch], n_nodes)

        # Needs to be numpy for measure libs infomeasure and Antropy
        pre_attack_S = np.array(base_S_hist)
        post_attack_S = np.array(pruned_S_hist_batch[batch])

        post_attack_W = take_submatrix(W0, take_idx)

        emsrs = entropic_measures(pre_attack_S, post_attack_S, take_idx)
        lz = lz_complexity_measures(pre_attack_S, post_attack_S, take_idx)
        sp = sample_entropy_measures(pre_attack_S, post_attack_S, take_idx)

        #post_G_reconstructed = nx.from_numpy_array(post_attack_W, create_using=nx.DiGraph)
        #gm = global_metrics_directed(post_G_reconstructed)

        gm = {
            "density": None,
            "global_efficiency": None,
            "avg_betweenness": None,
            "avg_closeness": None,
            "avg_clustering": None,
            "transitivity": None,
            "global_cc": None,
            "avg_spl": None,
            "swnss": None
        }
        result = {
            'batch': batch,
            'emsrs': emsrs,
            'lz': lz,
            'sp': sp,
            'gm': gm,
        }

        # Add driver fraction if I_ext is provided
        if I_ext is not None:
            b_driver_fraction = I_ext  # Replace with actual calculation if needed
            result['drf'] = {'batch_driver_fraction': b_driver_fraction[batch]}

        return result
    except Exception as e:
        logger.error(f"Error processing batch {batch}: {e}")
        return None

def flatten_metrics_dict(metrics_dict: Dict) -> Dict:
    """Flatten a nested dictionary of metrics into a single-level dictionary."""
    flattened = {}

    for category, metrics in metrics_dict.items():
        if category == 'batch':
            flattened['batch'] = metrics
            continue

        for metric_name, value in metrics.items():
            # Skip None values
            if value is None:
                continue

            # Format the key as category_metricname
            key = f"{category}_{metric_name}"
            flattened[key] = value

    return flattened

# Non-remote version for sequential processing
def process_batch_seq(batch: int, base_S_hist: np.ndarray, pruned_S_hist_batch: np.ndarray,
                 W0: np.ndarray, removed_ids: np.ndarray, n_nodes: int, I_ext: Optional[np.ndarray] = None) -> Dict:
    """
    Process a single batch - sequential version without Ray

    Args:
        batch: Batch index
        base_S_hist: Baseline spike history (pre-attack)
        pruned_S_hist_batch: Pruned spike history for this batch
        W0: Initial weight matrix
        removed_ids: Indices of removed nodes for this batch
        n_nodes: Total number of nodes in the original network
        I_ext: External input current (optional)

    Returns:
        Dictionary containing calculated metrics
    """
    try:
        take_idx = keep_indices(removed_ids[batch], n_nodes)

        # Needs to be numpy for measure libs infomeasure and Antropy
        pre_attack_S = np.array(base_S_hist)
        post_attack_S = np.array(pruned_S_hist_batch[batch])

        post_attack_W = take_submatrix(W0, take_idx)

        emsrs = entropic_measures(pre_attack_S, post_attack_S, take_idx)
        lz = lz_complexity_measures(pre_attack_S, post_attack_S, take_idx)
        sp = sample_entropy_measures(pre_attack_S, post_attack_S, take_idx)

        post_G_reconstructed = nx.from_numpy_array(post_attack_W, create_using=nx.DiGraph)
        gm = global_metrics_directed(post_G_reconstructed)

        result = {
            'batch': batch,
            'emsrs': emsrs,
            'lz': lz,
            'sp': sp,
            'gm': gm,
        }

        # Add driver fraction if I_ext is provided
        if I_ext is not None:
            b_driver_fraction = I_ext  # Replace with actual calculation if needed
            result['drf'] = {'batch_driver_fraction': b_driver_fraction[batch]}

        return result
    except Exception as e:
        logger.error(f"Error processing batch {batch}: {e}")
        return None

def process_directory(base_dir: Union[str, Path], n_workers: int = 4, force_sequential: bool = False) -> None:
    """
    Process all steps in a directory using Ray parallelization or sequential processing

    This function:
    1. Loads data from each step directory
    2. Processes each batch in parallel (using Ray) or sequentially
    3. Calculates resilience metrics for each batch
    4. Saves results as CSV files in each step directory

    Args:
        base_dir: Path to the base directory containing step directories
        n_workers: Number of Ray workers to use for parallel processing
        force_sequential: If True, use sequential processing instead of Ray
    """
    try:
        # Initialize Ray if we're not forcing sequential mode
        use_ray = not force_sequential
        ray_initialized = False

        if use_ray and not ray.is_initialized():
            try:
                ray.init(num_cpus=n_workers, ignore_reinit_error=True)
                logger.info(f"Ray initialized with {n_workers} workers")
                ray_initialized = True
            except Exception as e:
                logger.warning(f"Ray initialization error: {e}. Falling back to sequential processing.")
                use_ray = False

        # Validate base directory
        base_path = validate_base_dir(base_dir)
        logger.info(f"Processing directory: {base_path}")

        # Get available steps
        steps = list_available_steps(base_path)
        if not steps:
            logger.warning(f"No steps found in {base_path}")
            return

        logger.info(f"Found {len(steps)} steps to process")

        # Process each step
        for step in steps:
            step_dir = base_path / str(step)
            logger.info(f"Processing step {step} in {step_dir}")

             # Check if metrics.csv already exists
            metrics_path = step_dir / "metrics.csv"
            if metrics_path.exists():
                logger.info(f"Metrics file already exists for step {step}, skipping")
                continue

            # Load step data
            data = load_step_data(base_path, step)
            metadata = data['metadata']
            arrays = data['arrays']

            # Extract required arrays
            base_S_hist = arrays.get('base_S_hist')
            pruned_S_hist_batch = arrays.get('pruned_S_hist_batch')
            W0 = arrays.get('W0')
            removed_ids = arrays.get('removed_ids')
            batch_driver_fraction = arrays.get('batch_driver_fraction', None)

            if base_S_hist is None or pruned_S_hist_batch is None or W0 is None or removed_ids is None:
                logger.warning(f"Missing required arrays in step {step}, skipping")
                continue

            n_nodes = W0.shape[0]
            batch_size = pruned_S_hist_batch.shape[0]

            logger.info(f"Processing {batch_size} batches for step {step}")

            # Process batches (either with Ray or sequentially)
            if use_ray:
                # Create Ray tasks for each batch
                batch_futures = [
                    process_batch_ray.remote(
                        batch=i,
                        base_S_hist=base_S_hist,
                        pruned_S_hist_batch=pruned_S_hist_batch,
                        W0=W0,
                        removed_ids=removed_ids,
                        n_nodes=n_nodes,
                        I_ext=batch_driver_fraction
                    )
                    for i in range(batch_size)
                ]

                # Get results (this will wait for all tasks to complete)
                batch_results = ray.get(batch_futures)
            else:
                # Process sequentially
                logger.info(f"Processing {batch_size} batches sequentially")
                batch_results = []
                for i in range(batch_size):
                    logger.info(f"Processing batch {i+1}/{batch_size}")
                    result = process_batch_seq(
                        batch=i,
                        base_S_hist=base_S_hist,
                        pruned_S_hist_batch=pruned_S_hist_batch,
                        W0=W0,
                        removed_ids=removed_ids,
                        n_nodes=n_nodes,
                        I_ext=batch_driver_fraction
                    )
                    batch_results.append(result)

            # Filter out None results (from errors)
            batch_results = [r for r in batch_results if r is not None]

            if not batch_results:
                logger.warning(f"No valid results for step {step}, skipping")
                continue

            # Flatten metric dictionaries and create DataFrame
            flattened_results = [flatten_metrics_dict(result) for result in batch_results]
            df = pd.DataFrame(flattened_results)

            # Save results to CSV
            #metrics_path = step_dir / "metrics.csv"
            df.to_csv(metrics_path, index=False)
            logger.info(f"Saved metrics to {metrics_path}")

            # Create summary statistics
            summary_df = df.describe()
            summary_path = step_dir / "metrics_summary.csv"
            summary_df.to_csv(summary_path)
            logger.info(f"Saved summary statistics to {summary_path}")

    except Exception as e:
        logger.error(f"Error processing directory {base_dir}: {e}")
    finally:
        # Don't shut down Ray if we didn't initialize it
        if use_ray and ray_initialized and ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")

def main():
    """
    Main entry point for the script.

    Parses command line arguments and processes directories accordingly.

    Command line arguments:
        --base-dir: Path to the directory containing saved results
        --workers: Number of Ray workers to use for parallel processing
        --local: Run in sequential mode (no parallelism)
        --merge-only: Only merge existing metrics.csv files without recomputing metrics
    """
    parser = argparse.ArgumentParser(
        description="Extract metrics from attack results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all experiments in the 'save' directory with 8 workers
  python extract_metrics.py --base-dir save --workers 8

  # Process a specific experiment directory sequentially
  python extract_metrics.py --base-dir save/experiment_1 --local

  # Process all experiments with default settings (4 workers)
  python extract_metrics.py --base-dir save

  # Only merge existing metrics.csv files from step directories into a combined file
  python extract_metrics.py --base-dir save/experiment_1 --merge-only
"""
    )
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Base directory with saved results")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of Ray workers (default: 4)")
    parser.add_argument("--local", action="store_true",
                        help="Force sequential processing (no parallelism)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing metrics.csv files without recomputing metrics")
    args = parser.parse_args()

    # Check if sequential mode is requested
    force_sequential = args.local
    if force_sequential:
        logger.info("Running in sequential mode (no parallelism)")
        args.workers = 1

    # If base_dir points to a directory containing subdirectories, process each one
    base_path = Path(args.base_dir)
    if not base_path.exists():
        logger.error(f"Base directory does not exist: {base_path}")
        return

    if not base_path.is_dir():
        logger.error(f"Path is not a directory: {base_path}")
        return

    # Check if this is a step directory structure
    if any(item.is_dir() and item.name.isdigit() for item in base_path.iterdir()):
        # This is already a valid experiment directory
        if args.merge_only:
            merge_step_metrics(base_path)
        else:
            process_directory(base_path, args.workers, force_sequential)
    else:
        # This might be a container of multiple experiment directories
        # Look for subdirectories that contain step directories
        process_multiple_experiments(base_path, args.merge_only, args.workers, force_sequential)

def merge_step_metrics(save_dir: Union[str, Path]) -> None:
    """
    Merge metrics.csv files from all step directories into a single combined CSV file.

    This function:
    1. Identifies all step directories in save_dir (0, 1, 2, ...)
    2. Reads the metrics.csv file from each step directory
    3. Adds a 'step' column to identify which step each row came from
    4. Combines all data into a single DataFrame
    5. Saves the combined data as 'combined_metrics.csv' in save_dir

    Args:
        save_dir: Path to the directory containing step directories
    """
    try:
        # Validate save directory
        save_path = validate_base_dir(save_dir)
        logger.info(f"Merging metrics from step directories in: {save_path}")

        # Get available steps
        steps = list_available_steps(save_path)
        if not steps:
            logger.warning(f"No steps found in {save_path}")
            return

        logger.info(f"Found {len(steps)} steps to merge")

        # Initialize an empty list to store DataFrames from each step
        dfs = []

        # Process each step
        for step in steps:
            step_dir = save_path / str(step)
            metrics_path = step_dir / "metrics.csv"

            if not metrics_path.exists():
                logger.warning(f"No metrics.csv found for step {step}, skipping")
                continue

            # Read the metrics.csv file
            try:
                df = pd.read_csv(metrics_path)
                # Add a 'step' column to identify the source
                df['step'] = step
                dfs.append(df)
                logger.info(f"Loaded metrics from step {step} with {len(df)} rows")
            except Exception as e:
                logger.error(f"Error reading metrics from step {step}: {e}")

        if not dfs:
            logger.warning("No metrics data found in any step directory")
            return

        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)

        # Save the combined data
        output_path = save_path / "combined_metrics.csv"
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Saved combined metrics to {output_path} with {len(combined_df)} total rows")

    except Exception as e:
        logger.error(f"Error merging metrics from {save_dir}: {e}")


def process_multiple_experiments(base_dir: Union[str, Path], merge_only: bool = False, n_workers: int = 4, force_sequential: bool = False) -> None:
    """
    Process multiple experiment directories contained within a base directory.

    This function:
    1. Finds all subdirectories that contain step directories (0, 1, 2, ...)
    2. Processes each subdirectory using either merge_step_metrics or process_directory

    Args:
        base_dir: Path to the base directory containing experiment directories
        merge_only: If True, only merge existing metrics without computing them
        n_workers: Number of Ray workers to use (ignored if merge_only=True)
        force_sequential: If True, use sequential processing (ignored if merge_only=True)
    """
    base_path = validate_base_dir(base_dir)
    logger.info(f"Searching for experiment directories in: {base_path}")

    experiment_dirs_found = False

    # Look for subdirectories that contain step directories
    for subdir in base_path.iterdir():
        if not subdir.is_dir():
            continue

        # Check if this subdirectory contains steps
        if any(item.is_dir() and item.name.isdigit() for item in subdir.iterdir()):
            experiment_dirs_found = True
            logger.info(f"Found experiment directory: {subdir}")

            if merge_only:
                merge_step_metrics(subdir)
            else:
                process_directory(subdir, n_workers, force_sequential)

    if not experiment_dirs_found:
        logger.warning(f"No experiment directories found in {base_path}")

if __name__ == "__main__":
    main()

# Example command line usage:
# python extract_metrics.py --base-dir path/to/save/directory --workers 8
# python extract_metrics.py --base-dir path/to/save/directory --local  # Sequential mode
