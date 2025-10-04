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

import json
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
    global_metrics_directed,
    synchrony_measures
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
        sync = synchrony_measures(pre_attack_S, post_attack_S, take_idx, T=2000)

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
            'sync': sync
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
                 W0: np.ndarray, removed_ids: np.ndarray, n_nodes: int, I_ext: Optional[np.ndarray] = None, control: bool = False) -> Dict:
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
        if control:
            post_attack_S = np.array(pruned_S_hist_batch)[:, take_idx]
        else:
            post_attack_S = np.array(pruned_S_hist_batch[batch])



        #post_attack_W = take_submatrix(W0, take_idx)
        try:
            emsrs = entropic_measures(pre_attack_S, post_attack_S, take_idx)
        except Exception as e:
            logger.error(f"Error entropic_measures {batch}: {e}")
        #lz = lz_complexity_measures(pre_attack_S, post_attack_S, take_idx)
        lz = {
            "avg_lz_pre": None,
            "avg_lz_post_init": None,
            "avg_lz_post_final": None
        }
        sync = synchrony_measures(pre_attack_S, post_attack_S, take_idx, T=2000)

        sp = {
            "avg_se_pre": None,
            "avg_se_post_init": None,
            "avg_se_post_final": None
        }
        #sp = sample_entropy_measures(pre_attack_S, post_attack_S, take_idx)

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
        #post_G_reconstructed = nx.from_numpy_array(post_attack_W, create_using=nx.DiGraph)
        #gm = global_metrics_directed(post_G_reconstructed)

        result = {
            'batch': batch,
            'emsrs': emsrs,
            'lz': lz,
            'sp': sp,
            'gm': gm,
            'sync': sync
        }

        # Add driver fraction if I_ext is provided
        if I_ext is not None:
            b_driver_fraction = I_ext  # Replace with actual calculation if needed
            try:
                if control:
                    result['drf'] = {'batch_driver_fraction': b_driver_fraction[0]}
                else:
                    result['drf'] = {'batch_driver_fraction': b_driver_fraction[batch]}
            except Exception as e:
                logger.error(f"Error b_driver_fraction {batch}: {e}")


        return result
    except Exception as e:
        logger.error(f"Error processing batch {batch}: {e}")
        return None

def process_directory(base_dir: Union[str, Path], n_workers: int = 4, force_sequential: bool = False, control: bool = False) -> None:
    """
    Process all steps in a directory using Ray parallelization with pipelining

    This function:
    1. Loads data from each step directory
    2. Processes batches in a pipelined manner using Ray
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

        # For pipelining, we'll use a work queue approach
        if use_ray:
            # Track metadata
            step_metadata = {}

            for step in steps:
                step_dir = base_path / str(step)
                logger.info(f"Queueing step {step} in {step_dir}")

                # Check if metrics.csv already exists
                metrics_path = step_dir / "metrics.csv"
                if metrics_path.exists():
                    logger.info(f"Metrics file already exists for step {step}, skipping")
                    continue

                # Load step data
                data = load_step_data(base_path, step)
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

                logger.info(f"Queueing {batch_size} batches for step {step}")

                # Put large arrays into Ray object store ONCE
                base_S_hist_ref = ray.put(base_S_hist)
                pruned_S_hist_batch_ref = ray.put(pruned_S_hist_batch)
                W0_ref = ray.put(W0)
                removed_ids_ref = ray.put(removed_ids)
                batch_driver_fraction_ref = ray.put(batch_driver_fraction)

                # Limit inflight tasks to avoid memory / spillover
                max_inflight = n_workers * 2
                futures = []
                results = []

                for i in range(batch_size):
                    futures.append(
                        process_batch_ray.remote(
                            batch=i,
                            base_S_hist=base_S_hist_ref,
                            pruned_S_hist_batch=pruned_S_hist_batch_ref,
                            W0=W0_ref,
                            removed_ids=removed_ids_ref,
                            n_nodes=n_nodes,
                            I_ext=batch_driver_fraction_ref
                        )
                    )

                    if len(futures) >= max_inflight:
                        done, futures = ray.wait(futures, num_returns=1)
                        results.extend(ray.get(done))

                # collect any remaining results
                if futures:
                    results.extend(ray.get(futures))

                # Filter out None results
                batch_results = [r for r in results if r is not None]

                if not batch_results:
                    logger.warning(f"No valid results for step {step}, skipping")
                    continue

                # Flatten metric dictionaries and create DataFrame
                flattened_results = [flatten_metrics_dict(result) for result in batch_results]
                df = pd.DataFrame(flattened_results)

                # Save results to CSV
                df.to_csv(metrics_path, index=False)
                logger.info(f"Saved metrics to {metrics_path}")

                # Create summary statistics
                summary_df = df.describe()
                summary_path = step_dir / "metrics_summary.csv"
                summary_df.to_csv(summary_path)
                logger.info(f"Saved summary statistics to {summary_path}")

        else:
            # Sequential processing (original code)
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
                batch_size = removed_ids.shape[0]

                logger.info(f"Processing {batch_size} batches for step {step} sequentially")
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
                        I_ext=batch_driver_fraction,
                        control=control
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
                df.to_csv(metrics_path, index=False)
                logger.info(f"Saved metrics to {metrics_path}")

                # Create summary statistics
                summary_df = df.describe()
                summary_path = step_dir / "metrics_summary.csv"
                summary_df.to_csv(summary_path)
                logger.info(f"Saved summary statistics to {summary_path}")

                G_W0 = nx.from_numpy_array(W0, create_using=nx.DiGraph)
                gm = global_metrics_directed(G_W0)

                # Save graph metrics to JSON
                graph_metrics_path = step_dir / "graph_metrics.json"
                with open(graph_metrics_path, 'w') as f:
                    json.dump(gm, f, indent=4)
                logger.info(f"Saved graph metrics to {graph_metrics_path}")


    except Exception as e:
        logger.error(f"Error processing directory {base_dir}: {e}")
    finally:
        # Don't shut down Ray if we didn't initialize it
        if use_ray and ray_initialized and ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")

def _process_directory(base_dir: Union[str, Path], n_workers: int = 4, force_sequential: bool = False) -> None:
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
    parser.add_argument("--control", action='store_true', default=False,
                        help="Run control metrics mode")

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

    control = bool(args.control)
    # Check if this is a step directory structure
    if any(item.is_dir() and item.name.isdigit() for item in base_path.iterdir()):
        # This is already a valid experiment directory
        if args.merge_only:
            merge_step_metrics(base_path)
        else:
            process_directory(base_path, args.workers, force_sequential, control)
    else:
        # This might be a container of multiple experiment directories
        # Look for subdirectories that contain step directories
        process_multiple_experiments(base_path, args.merge_only, args.workers, force_sequential)


def merge_step_metrics(save_dir: Union[str, Path]) -> None:
    """
    Merge metrics.csv files from all step directories into a single combined CSV file.
    Also merge graph_metrics.json files into a separate combined CSV file.

    This function:
    1. Identifies all step directories in save_dir (0, 1, 2, ...)
    2. Reads the metrics.csv file from each step directory
    3. Reads the graph_metrics.json file from each step directory
    4. Adds a 'step' column to identify which step each row came from
    5. Combines all metrics data into a single DataFrame
    6. Combines all graph metrics data into a separate single DataFrame
    7. Saves the combined data as 'combined_metrics.csv' and 'combined_graph_metrics.csv' in save_dir

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

        # Initialize empty lists to store DataFrames from each step
        metric_dfs = []
        graph_metric_dfs = []

        # Process each step
        for step in steps:
            step_dir = save_path / str(step)

            # Process metrics.csv
            metrics_path = step_dir / "metrics.csv"
            if metrics_path.exists():
                try:
                    df = pd.read_csv(metrics_path)
                    # Add a 'step' column to identify the source
                    df['step'] = step
                    metric_dfs.append(df)
                    logger.info(f"Loaded metrics from step {step} with {len(df)} rows")
                except Exception as e:
                    logger.error(f"Error reading metrics from step {step}: {e}")
            else:
                logger.warning(f"No metrics.csv found for step {step}, skipping")

            # Process graph_metrics.json
            graph_metrics_path = step_dir / "graph_metrics.json"
            if graph_metrics_path.exists():
                try:
                    with open(graph_metrics_path, 'r') as f:
                        graph_metrics = json.load(f)

                    # Convert JSON to DataFrame (single row)
                    graph_df = pd.DataFrame([graph_metrics])
                    # Add a 'step' column to identify the source
                    graph_df['step'] = step
                    graph_metric_dfs.append(graph_df)
                    logger.info(f"Loaded graph metrics from step {step}")
                except Exception as e:
                    logger.error(f"Error reading graph metrics from step {step}: {e}")
            else:
                logger.warning(f"No graph_metrics.json found for step {step}, skipping")

        # Combine and save metrics.csv data
        if metric_dfs:
            combined_df = pd.concat(metric_dfs, ignore_index=True)
            output_path = save_path / "combined_metrics.csv"
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Saved combined metrics to {output_path} with {len(combined_df)} total rows")
        else:
            logger.warning("No metrics data found in any step directory")

        # Combine and save graph_metrics.json data
        if graph_metric_dfs:
            combined_graph_df = pd.concat(graph_metric_dfs, ignore_index=True)
            output_path = save_path / "combined_graph_metrics.csv"
            combined_graph_df.to_csv(output_path, index=False)
            logger.info(f"Saved combined graph metrics to {output_path} with {len(combined_graph_df)} total rows")
        else:
            logger.warning("No graph metrics data found in any step directory")

    except Exception as e:
        logger.error(f"Error merging metrics from {save_dir}: {e}")


def _merge_step_metrics(save_dir: Union[str, Path]) -> None:
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
