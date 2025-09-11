from networkx.readwrite.json_graph import adjacency
import orbax.checkpoint as ocp
import jax
import numpy as np
from jax import numpy as jnp
from pathlib import Path
from attack import keep_indices
import matplotlib.pyplot as plt
from measures import entropic_measures,lz_complexity_measures, sample_entropy_measures, global_metrics_directed
from utils import plot_raster_simple
from tqdm import tqdm
import networkx as nx
import time
from importlib import metadata
import pandas as pd
import logging
import concurrent.futures
import multiprocessing
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_save_dirs(base_dir_path: str) -> list:
    """
    Returns a list of all subdirectories in the given base directory.

    Args:
        base_dir_path: Path to the base directory to scan for subdirectories

    Returns:
        List of Path objects representing subdirectories
    """
    base_dir = Path(base_dir_path).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # Get all subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]

    return subdirs

def take_submatrix(M: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return M[np.ix_(idx, idx)]

def process_batch(batch, base_S_hist, pruned_S_hist_batch, W0, removed_ids, n_nodes, I_ext):
    """
    Process a single batch - modified for parallel execution
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

        #mean_abs_syn = np.mean(np.abs(pruned_syn_hist[batch]))      # average across time & neurons
        #b_driver_fraction = mean_abs_syn / (mean_abs_syn + I_ext)

        return {
            'batch': batch,
            'emsrs': emsrs,
            'lz': lz,
            'sp': sp,
            'gm': gm,
       #     'drf': {'batch_driver_fraction': b_driver_fraction[batch]}
        }
    except Exception as e:
        logger.error(f"Error processing batch {batch}: {e}")
        return None

def process_single_step(step, base_dir, batch_size=20, I_ext=10.0, max_workers=None):
    """Process a single step with parallel batch processing"""
    logger.info(f"Processing step {step}")

    # Set default max_workers to number of CPU cores
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    # Create analysis directory for this step
    analysis_dir = base_dir / str(step)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = analysis_dir / 'metrics.csv'

    # Skip if already processed
    if analysis_path.exists():
        logger.info(f"Step {step} already processed. Skipping...")
        return True

    try:
        with ocp.CheckpointManager(base_dir) as mngr:
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
            # Calculate base metrics
            pre_G_reconstructed = nx.from_numpy_array(W0, create_using=nx.DiGraph)
            gm0 = global_metrics_directed(pre_G_reconstructed)

            #base_syn_hist = restored.arrays['base_syn_hist']
            #mean_abs_syn = np.mean(np.abs(base_syn_hist))
            #base_driver_fraction = mean_abs_syn / (mean_abs_syn + I_ext)

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
                #b_driver_fraction=b_driver_fraction,
                n_nodes=n_nodes,
                I_ext=I_ext
            )

            # Process batches in parallel
            logger.info(f"Processing {batch_size} batches using {max_workers} workers")

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batch jobs
                batch_indices = list(range(batch_size))
                future_to_batch = {
                    executor.submit(process_batch_partial, batch): batch
                    for batch in batch_indices
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

            time_end = time.time()
            logger.info(f"Step {step} processing time: {time_end - time_start:.2f} seconds")
            logger.info(f"Successfully processed {len(results)}/{batch_size} batches")

            if not results:
                logger.error(f"No batches were successfully processed for step {step}")
                return False

            # Collect results
            all_emsrs = [r['emsrs'] for r in results]
            all_lz = [r['lz'] for r in results]
            all_sp = [r['sp'] for r in results]
            all_gm = [r['gm'] for r in results]
            #all_drf = [r['drf'] for r in results]

            # Create DataFrame
            all_metrics = [all_emsrs, all_lz, all_sp, all_gm]
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

            return True

    except Exception as e:
        logger.error(f"Error processing step {step}: {e}")
        return False

def create_combined_metrics(base_dir, successful_steps):
    """Create a combined metrics file from all successful steps"""
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

def main(max_workers=None, batch_size=20, I_ext=10.0, base_dir=None):
    """Main function to process all steps with parallel batch processing"""

    # Example usage
    if base_dir is None:
        base_dir = Path("save/save_test_ER_dense_stdp").resolve()
        base_dir.mkdir(parents=True, exist_ok=True)

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

    total_end_time = time.time()

    # Summary
    logger.info("="*50)
    logger.info("PROCESSING COMPLETE!")
    logger.info(f"Total time: {total_end_time - total_start_time:.2f} seconds")
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

if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--base_dir', help='Base directory to process')
    def cli(base_dir):
        if base_dir:
            # Use the specified base directory
            successful, failed = main(base_dir=Path(base_dir))
        else:
            # Process all directories in 'save'
            all_base_dirs = get_save_dirs('save')
            for base_dir in all_base_dirs:
                successful, failed = main(base_dir=base_dir)

    cli()
