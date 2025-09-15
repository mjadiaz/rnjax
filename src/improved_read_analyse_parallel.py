#!/usr/bin/env python3
"""
Improved version of read_analyse_parallel.py

This script provides functionality for analyzing neural network checkpoints
in parallel, processing batches of data to calculate various complexity and
network metrics.
"""
import concurrent.futures
import logging
import multiprocessing
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import click
import jax
import networkx as nx
import numpy as np
import pandas as pd
from jax import numpy as jnp
import orbax.checkpoint as ocp
from tqdm import tqdm

# Local imports
from attack import keep_indices
from measures import entropic_measures, lz_complexity_measures, sample_entropy_measures, global_metrics_directed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# ===== Utility Functions =====

def get_save_dirs(base_dir_path: str) -> List[Path]:
    """
    Returns a list of all subdirectories in the given base directory.

    Args:
        base_dir_path: Path to the base directory to scan for subdirectories

    Returns:
        List of Path objects representing subdirectories
    """
    base_dir = Path(base_dir_path).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    return [d for d in base_dir.iterdir() if d.is_dir()]


def take_submatrix(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Extract a submatrix using the given indices.

    Args:
        matrix: The original matrix
        indices: Indices to keep

    Returns:
        Submatrix containing only the specified indices
    """
    return matrix[np.ix_(indices, indices)]


# ===== Core Processing Functions =====

def process_batch(
    batch_idx: int,
    base_S_hist: np.ndarray,
    pruned_S_hist_batch: np.ndarray,
    W0: np.ndarray,
    removed_ids: np.ndarray,
    n_nodes: int,
    I_ext: float
) -> Optional[Dict[str, Any]]:
    """
    Process a single batch - modified for parallel execution

    Args:
        batch_idx: Index of the batch to process
        base_S_hist: Base spike history
        pruned_S_hist_batch: Pruned spike history for all batches
        W0: Weight matrix
        removed_ids: IDs of removed neurons for all batches
        n_nodes: Number of nodes
        I_ext: External input current

    Returns:
        Dictionary containing processed metrics or None if processing failed
    """
    try:
        # Get indices to keep after pruning
        take_idx = keep_indices(removed_ids[batch_idx], n_nodes)

        # Convert to numpy for measure libraries
        pre_attack_S = np.array(base_S_hist)
        post_attack_S = np.array(pruned_S_hist_batch[batch_idx])

        # Extract the submatrix corresponding to remaining neurons
        post_attack_W = take_submatrix(W0, take_idx)

        # Calculate various metrics
        emsrs = entropic_measures(pre_attack_S, post_attack_S, take_idx)
       # lz = lz_complexity_measures(pre_attack_S, post_attack_S, take_idx)
       # sp = sample_entropy_measures(pre_attack_S, post_attack_S, take_idx)

        # Network analysis
        post_G_reconstructed = nx.from_numpy_array(post_attack_W, create_using=nx.DiGraph)
        gm = global_metrics_directed(post_G_reconstructed)

        return {
            'batch': batch_idx,
            'emsrs': emsrs,
         #   'lz': lz,
       #     'sp': sp,
            'gm': gm,
        }
    except Exception as e:
        logger.error(f"Error processing batch {batch_idx}: {e}")
        return None


def process_single_step(
    step: int,
    base_dir: Path,
    batch_size: int = 20,
    I_ext: float = 10.0,
    max_workers: Optional[int] = None
) -> bool:
    """
    Process a single checkpoint step with parallel batch processing

    Args:
        step: Step number to process
        base_dir: Base directory containing checkpoints
        batch_size: Expected batch size
        I_ext: External input current
        max_workers: Maximum number of parallel workers (defaults to CPU count)

    Returns:
        True if processing succeeded, False otherwise
    """
    logger.info(f"Processing step {step}")

    # Set default max_workers to number of CPU cores
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    # Create analysis directory for this step
    analysis_dir = base_dir / str(step)
    absolute_base_dir = base_dir.resolve()
    #analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_dir / 'metrics.csv'

    # Skip if already processed
    if analysis_path.exists():
        logger.info(f"Step {step} already processed. Skipping...")
        return True

    try:
        with ocp.CheckpointManager(absolute_base_dir) as mngr:
            time_start = time.time()

            # Try to restore the checkpoint for this step
            try:
                restored = mngr.restore(step)
            except Exception as e:
                logger.warning(f"Could not restore checkpoint for step {step}: {e}")
                return False

            logger.info(f"Restored arrays: {list(restored.arrays.keys())}")

            # Extract data
            base_S_hist = restored.arrays['base_S_hist']
            pruned_S_hist_batch = restored.arrays['pruned_S_hist_batch']
            W0 = np.array(restored.arrays['W0'])
            removed_ids = restored.arrays['removed_ids']
            neuron_type = restored.arrays['neuron_type']
            n_nodes = restored.metadata['parameters']['N']
            base_driver_fraction = restored.arrays['base_driver_fraction']
            batch_driver_fraction = restored.arrays['batch_driver_fraction']

            # Calculate base metrics for original network
            pre_G_reconstructed = nx.from_numpy_array(W0, create_using=nx.DiGraph)
            gm0 = global_metrics_directed(pre_G_reconstructed)

            # Get actual batch size from data
            actual_batch_size = pruned_S_hist_batch.shape[0]
            if actual_batch_size != batch_size:
                logger.warning(f"Expected batch size {batch_size}, got {actual_batch_size}")
                batch_size = actual_batch_size

            # Create partial function with fixed arguments for parallel processing
            process_batch_partial = partial(
                process_batch,
                base_S_hist=base_S_hist,
                pruned_S_hist_batch=pruned_S_hist_batch,
                W0=W0,
                removed_ids=removed_ids,
                n_nodes=n_nodes,
                I_ext=I_ext
            )

            # Process batches in parallel
            logger.info(f"Processing {batch_size} batches using {max_workers} workers")

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batch jobs
                batch_indices = list(range(batch_size))
                future_to_batch = {
                    executor.submit(process_batch_partial, batch_idx): batch_idx
                    for batch_idx in batch_indices
                }

                # Collect results with progress bar
                results = []
                failed_batches = []

                with tqdm(total=batch_size, desc=f"Processing batches for step {step}") as pbar:
                    for future in concurrent.futures.as_completed(future_to_batch):
                        batch = future_to_batch[future]
                        try:
                            result = future.result()
                            if result is not None:
                                results.append(result)
                            else:
                                failed_batches.append(batch)
                        except Exception as e:
                            logger.error(f"Error processing batch {batch} in step {step}: {e}")
                            failed_batches.append(batch)
                        pbar.update(1)

                if failed_batches:
                    logger.warning(f"Failed batches in step {step}: {failed_batches}")

            # Sort results by batch number to maintain order
            results.sort(key=lambda x: x['batch'])

            processing_time = time.time() - time_start
            logger.info(f"Step {step} processing time: {processing_time:.2f} seconds")
            logger.info(f"Successfully processed {len(results)}/{batch_size} batches")

            if not results:
                logger.error(f"No batches were successfully processed for step {step}")
                return False

            # Collect results
            all_emsrs = [r['emsrs'] for r in results]
            #all_lz = [r['lz'] for r in results]
            #all_sp = [r['sp'] for r in results]
            all_gm = [r['gm'] for r in results]

            # Create DataFrame from results
            all_metrics = [
                all_emsrs,
                #all_lz,
                #all_sp,
                all_gm]
            collect_rows = []

            for b_idx, result in enumerate(results):
                # Merge all metrics for this batch
                all_metrics_batch = [all_metrics[m][b_idx] for m in range(len(all_metrics))]
                merged = {k: v for d in all_metrics_batch for k, v in d.items()}

                # Add metadata
                merged['step'] = step
                merged['batch_id'] = result['batch']
                collect_rows.append(merged)

            df = pd.DataFrame(collect_rows)

            # Add base network metrics with _pre suffix
            gm0_renamed = {key + "_pre": value for key, value in gm0.items()}
            for col, val in gm0_renamed.items():
                df[col] = val

            # Add driver fractions
            df['base_driver_fraction'] = np.unique(base_driver_fraction)[0]
            df['batch_driver_fraction'] = batch_driver_fraction

            # Save to CSV
            df.to_csv(analysis_path, index=False)
            logger.info(f"Saved metrics for step {step} to {analysis_path}")

            return True

    except Exception as e:
        logger.error(f"Error processing step {step}: {e}")
        return False


def create_combined_metrics(base_dir: Path, successful_steps: List[int]) -> None:
    """
    Create a combined metrics file from all successful steps

    Args:
        base_dir: Base directory containing step subdirectories
        successful_steps: List of step numbers that were successfully processed
    """
    combined_data = []

    logger.info("Creating combined metrics file...")

    for step in successful_steps:
        metrics_path = base_dir / str(step) / 'metrics.csv'
        if metrics_path.exists():
            try:
                df_step = pd.read_csv(metrics_path)
                combined_data.append(df_step)
                logger.info(f"Added step {step} to combined metrics")
            except Exception as e:
                logger.error(f"Error reading metrics for step {step}: {e}")

    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_path = base_dir / 'combined_metrics.csv'
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"Created combined metrics file: {combined_path}")
        logger.info(f"Combined file contains {len(combined_df)} rows from {len(successful_steps)} steps")
    else:
        logger.warning("No data available for combined metrics file")


def resave_to_host(base_dir: Path) -> Tuple[List[int], List[int]]:
    """
    Check if checkpoint data is on GPU then resave to host

    Args:
        base_dir: Base directory containing checkpoints

    Returns:
        Tuple of (successful_steps, failed_steps)
    """
    base_dir = Path(base_dir).resolve()
    n_of_directories = len([d for d in base_dir.iterdir() if d.is_dir()])

    logger.info(f"Found {n_of_directories} directories to process")

    successful_steps = []
    failed_steps = []

    # Process each step
    for step in tqdm(range(n_of_directories)):
        logger.info(f"Processing step {step}")

        try:
            with ocp.CheckpointManager(base_dir) as mngr:
                time_start = time.time()

                # Try to restore the checkpoint for this step
                try:
                    restored_state = mngr.restore(step)
                except Exception as e:
                    logger.warning(f"Could not restore checkpoint for step {step}: {e}")
                    failed_steps.append(step)
                    continue

                logger.info(f"Restored arrays: {list(restored_state.arrays.keys())}")

                # Check if data is on GPU and move to host
                arrays_on_gpu = False
                for key, array in restored_state.arrays.items():
                    # Check if array is on GPU device
                    if hasattr(array, 'device') and 'gpu' in str(array.device).lower():
                        arrays_on_gpu = True
                        logger.info(f"Array '{key}' found on GPU device: {array.device}")
                    elif hasattr(array, 'devices') and any('gpu' in str(device).lower() for device in array.devices()):
                        arrays_on_gpu = True
                        logger.info(f"Array '{key}' found on GPU devices: {array.devices()}")

                if arrays_on_gpu:
                    logger.info(f"Moving data from GPU to host for step {step}")
                    # Move all arrays to host
                    state_cpu = jax.device_get(restored_state)

                    # Create a new checkpoint directory for CPU version
                    cpu_base_dir = base_dir.parent / (base_dir.name + "_cpu")
                    cpu_base_dir.mkdir(parents=True, exist_ok=True)

                    # Save the CPU version
                    with ocp.CheckpointManager(cpu_base_dir) as cpu_mngr:
                        cpu_mngr.save(step, state_cpu)
                        logger.info(f"Saved CPU version of step {step} to {cpu_base_dir}")

                    successful_steps.append(step)
                else:
                    logger.info(f"Step {step} data already on host/CPU")
                    successful_steps.append(step)

                processing_time = time.time() - time_start
                logger.info(f"Step {step} processing time: {processing_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error processing step {step}: {e}")
            failed_steps.append(step)

    # Summary
    logger.info("="*50)
    logger.info("RESAVE TO HOST COMPLETE!")
    logger.info(f"Successful steps ({len(successful_steps)}): {successful_steps}")
    if failed_steps:
        logger.warning(f"Failed steps ({len(failed_steps)}): {failed_steps}")
    else:
        logger.info("All steps processed successfully!")
    logger.info("="*50)

    return successful_steps, failed_steps


def main_sequential(
    base_dir,
    batch_size: int = 20,
    I_ext: float = 10.0,

) -> Tuple[List[int], List[int]]:
    """
    Main function to process all steps sequentially (no parallel processing)

    Args:
        batch_size: Number of batches per step
        I_ext: External input current
        base_dir: Base directory containing checkpoints

    Returns:
        Tuple of (successful_steps, failed_steps)
    """
    # Set default base directory if not provided

    absolute_base_dir = base_dir.resolve()


    n_of_directories = len([d for d in base_dir.iterdir() if d.is_dir()])

    logger.info(f"Found {n_of_directories} directories to process")
    logger.info(f"Processing sequentially (no parallelization)")
    logger.info(f"Batch size: {batch_size}, I_ext: {I_ext}")

    successful_steps = []
    failed_steps = []

    total_start_time = time.time()

    # Process each step
    for step in range(n_of_directories):
        logger.info(f"Processing step {step}")

        # Create analysis directory for this step
        analysis_dir = base_dir / str(step)
        #analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = analysis_dir / 'metrics.csv'

        # Skip if already processed
        if analysis_path.exists():
            logger.info(f"Step {step} already processed. Skipping...")
            successful_steps.append(step)
            continue

        try:
            with ocp.CheckpointManager(absolute_base_dir) as mngr:
                time_start = time.time()

                # Try to restore the checkpoint for this step
                try:
                    restored = mngr.restore(step)
                except Exception as e:
                    logger.warning(f"Could not restore checkpoint for step {step}: {e}")
                    failed_steps.append(step)
                    continue

                logger.info(f"Restored arrays: {list(restored.arrays.keys())}")

                # Extract data
                base_S_hist = restored.arrays['base_S_hist']
                pruned_S_hist_batch = restored.arrays['pruned_S_hist_batch']
                W0 = np.array(restored.arrays['W0'])
                removed_ids = restored.arrays['removed_ids']
                neuron_type = restored.arrays['neuron_type']
                n_nodes = restored.metadata['parameters']['N']
                base_driver_fraction = restored.arrays['base_driver_fraction']
                batch_driver_fraction = restored.arrays['batch_driver_fraction']

                # Calculate base metrics
                pre_G_reconstructed = nx.from_numpy_array(W0, create_using=nx.DiGraph)
                gm0 = global_metrics_directed(pre_G_reconstructed)

                # Get actual batch size from data
                actual_batch_size = pruned_S_hist_batch.shape[0]
                if actual_batch_size != batch_size:
                    logger.warning(f"Expected batch size {batch_size}, got {actual_batch_size}")
                    batch_size = actual_batch_size

                # Process batches sequentially
                logger.info(f"Processing {batch_size} batches sequentially")

                results = []
                failed_batches = []

                # Process each batch one by one with progress bar
                with tqdm(total=batch_size, desc=f"Processing batches for step {step}") as pbar:
                    for batch in range(batch_size):
                        try:
                            result = process_batch(
                                batch_idx=batch,
                                base_S_hist=base_S_hist,
                                pruned_S_hist_batch=pruned_S_hist_batch,
                                W0=W0,
                                removed_ids=removed_ids,
                                n_nodes=n_nodes,
                                I_ext=I_ext
                            )
                            if result is not None:
                                results.append(result)
                            else:
                                failed_batches.append(batch)
                        except Exception as e:
                            logger.error(f"Error processing batch {batch} in step {step}: {e}")
                            failed_batches.append(batch)
                        pbar.update(1)

                if failed_batches:
                    logger.warning(f"Failed batches in step {step}: {failed_batches}")

                processing_time = time.time() - time_start
                logger.info(f"Step {step} processing time: {processing_time:.2f} seconds")
                logger.info(f"Successfully processed {len(results)}/{batch_size} batches")

                if not results:
                    logger.error(f"No batches were successfully processed for step {step}")
                    failed_steps.append(step)
                    continue

                # Collect results
                all_emsrs = [r['emsrs'] for r in results]
                #all_lz = [r['lz'] for r in results]
                #all_sp = [r['sp'] for r in results]
                all_gm = [r['gm'] for r in results]

                # Create DataFrame
                all_metrics = [
                    all_emsrs,
                    #all_lz,
                    #all_sp,
                    all_gm
                ]
                collect_rows = []

                for b in range(len(results)):
                    all_metrics_batch = [all_metrics[m][b] for m in range(len(all_metrics))]
                    merged = {k: v for d in all_metrics_batch for k, v in d.items()}
                    merged['step'] = step  # Add step information
                    merged['batch_id'] = results[b]['batch']  # Add original batch ID
                    collect_rows.append(merged)

                df = pd.DataFrame(collect_rows)

                # Add base repeated quantities
                gm0_renamed = {key + "_pre": value for key, value in gm0.items()}
                for col, val in gm0_renamed.items():
                    df[col] = val

                df['base_driver_fraction'] = np.unique(base_driver_fraction)[0]
                df['batch_driver_fraction'] = batch_driver_fraction

                # Save to CSV
                df.to_csv(analysis_path, index=False)
                logger.info(f"Saved metrics for step {step} to {analysis_path}")

                successful_steps.append(step)

        except Exception as e:
            logger.error(f"Error processing step {step}: {e}")
            failed_steps.append(step)

    total_processing_time = time.time() - total_start_time

    # Summary
    logger.info("="*50)
    logger.info("SEQUENTIAL PROCESSING COMPLETE!")
    logger.info(f"Total time: {total_processing_time:.2f} seconds")
    logger.info(f"Successful steps ({len(successful_steps)}): {successful_steps}")
    if failed_steps:
        logger.warning(f"Failed steps ({len(failed_steps)}): {failed_steps}")
    else:
        logger.info("All steps processed successfully!")
    logger.info("="*50)

    # Create combined metrics file
    if successful_steps:
        create_combined_metrics(base_dir, successful_steps)

    return successful_steps, failed_steps


def main_parallel(
    base_dir, #str or Path
    max_workers: Optional[int] = None,
    batch_size: int = 20,
    I_ext: float = 10.0,
) -> Tuple[List[int], List[int]]:
    """
    Main function to process all steps with parallel batch processing

    Args:
        max_workers: Maximum number of parallel workers (defaults to CPU count)
        batch_size: Number of batches per step
        I_ext: External input current
        base_dir: Base directory containing checkpoints

    Returns:
        Tuple of (successful_steps, failed_steps)
    """

    n_of_directories = len([d for d in base_dir.iterdir() if d.is_dir()])

    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    logger.info(f"Found {n_of_directories} directories to process")
    logger.info(f"Using {max_workers} workers for parallel batch processing")
    logger.info(f"Batch size: {batch_size}, I_ext: {I_ext}")

    successful_steps = []
    failed_steps = []

    total_start_time = time.time()

    # Process each step
    for step in range(n_of_directories):
        success = process_single_step(
            step=step,
            base_dir=base_dir,
            batch_size=batch_size,
            I_ext=I_ext,
            max_workers=max_workers
        )
        if success:
            successful_steps.append(step)
        else:
            failed_steps.append(step)

    total_processing_time = time.time() - total_start_time

    # Summary
    logger.info("="*50)
    logger.info("PARALLEL PROCESSING COMPLETE!")
    logger.info(f"Total time: {total_processing_time:.2f} seconds")
    logger.info(f"Successful steps ({len(successful_steps)}): {successful_steps}")
    if failed_steps:
        logger.warning(f"Failed steps ({len(failed_steps)}): {failed_steps}")
    else:
        logger.info("All steps processed successfully!")
    logger.info("="*50)

    # Create combined metrics file
    if successful_steps:
        create_combined_metrics(base_dir, successful_steps)

    return successful_steps, failed_steps


# ===== Command Line Interface =====

@click.group(help="Neural network checkpoint analysis tools")
def cli():
    """Main CLI entry point with command groups"""
    pass


@cli.command("process", help="Process checkpoint data with various options")
@click.option(
    '--base-dir', '-d',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help='Base directory containing checkpoints'
)
@click.option(
    '--step', '-s',
    type=int,
    help='Specific step to process (processes all steps if not specified)'
)
@click.option(
    '--batch-size', '-b',
    type=int,
    default=20,
    help='Batch size (default: 20)'
)
@click.option(
    '--i-ext', '-i',
    type=float,
    default=10.0,
    help='External current value (default: 10.0)'
)
@click.option(
    '--parallel/--sequential',
    default=True,
    help='Use parallel processing (default) or sequential processing'
)
@click.option(
    '--workers', '-w',
    type=int,
    default=None,
    help='Maximum number of workers for parallel processing (default: CPU count)'
)
def process_command(base_dir, step, batch_size, i_ext, parallel, workers):
    """Process checkpoint data with various options"""
    if base_dir is None:
        # Process all directories in 'save' if no base_dir specified
        all_base_dirs = get_save_dirs('save')
        if not all_base_dirs:
            logger.error("No directories found in 'save' folder")
            return

        for dir_path in all_base_dirs:
            logger.info(f"Processing directory: {dir_path}")
            _process_dir(dir_path, step, batch_size, i_ext, parallel, workers)
    else:
        _process_dir(base_dir, step, batch_size, i_ext, parallel, workers)


def _process_dir(base_dir, step, batch_size, i_ext, parallel, workers):
    """Helper function to process a single directory"""
    if step is not None:
        # Process single step
        success = process_single_step(
            step=step,
            base_dir=base_dir,
            batch_size=batch_size,
            I_ext=i_ext,
            max_workers=workers if parallel else 1
        )
        logger.info(f"Processing step {step}: {'Success' if success else 'Failed'}")
    else:
        # Process all steps
        if parallel:
            successful, failed = main_parallel(
                base_dir=base_dir,
                batch_size=batch_size,
                I_ext=i_ext,
                max_workers=workers
            )
        else:
            successful, failed = main_sequential(
                base_dir=base_dir,
                batch_size=batch_size,
                I_ext=i_ext
            )


@cli.command("resave", help="Move GPU data to host/CPU and resave")
@click.option(
    '--base-dir', '-d',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help='Base directory containing checkpoints'
)
def resave_command(base_dir):
    """Resave checkpoint data from GPU to CPU"""
    if base_dir is None:
        # Process all directories in 'save' if no base_dir specified
        all_base_dirs = get_save_dirs('save')
        if not all_base_dirs:
            logger.error("No directories found in 'save' folder")
            return

        for dir_path in all_base_dirs:
            logger.info(f"Resaving directory: {dir_path}")
            successful, failed = resave_to_host(base_dir=dir_path)
    else:
        successful, failed = resave_to_host(base_dir=base_dir)


@cli.command("combine", help="Combine metrics from multiple steps")
@click.option(
    '--base-dir', '-d',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help='Base directory containing step subdirectories with metrics.csv files'
)
def combine_command(base_dir):
    """Combine metrics from multiple steps into a single file"""
    # Find all step directories that have metrics.csv
    steps_with_metrics = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            metrics_file = item / 'metrics.csv'
            if metrics_file.exists():
                steps_with_metrics.append(int(item.name))

    if not steps_with_metrics:
        logger.error(f"No step directories with metrics.csv found in {base_dir}")
        return

    logger.info(f"Found {len(steps_with_metrics)} steps with metrics files")
    create_combined_metrics(base_dir, steps_with_metrics)


if __name__ == "__main__":
    # Show version info at startup
    try:
        logger.info(f"Python {multiprocessing.cpu_count()}-core environment")
    except Exception:
        pass

    # Run CLI
    cli()
