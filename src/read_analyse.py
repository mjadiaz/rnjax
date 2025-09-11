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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_dir = Path("save_test_ER_dense_stdp").resolve()
base_dir.mkdir(parents=True, exist_ok=True)

n_of_directories = len([d for d in base_dir.iterdir() if d.is_dir()])

def take_submatrix(M: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return M[np.ix_(idx, idx)]

def process_batch(batch, base_S_hist, pruned_S_hist_batch, W0, removed_ids, pruned_syn_hist, n_nodes, I_ext):
    """Process a single batch"""
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

    mean_abs_syn = np.mean(np.abs(pruned_syn_hist[batch]))
    b_driver_fraction = mean_abs_syn / (mean_abs_syn + I_ext)

    return {
        'batch': batch,
        'emsrs': emsrs,
        'lz': lz,
        'sp': sp,
        'gm': gm,
        'drf': {'batch_driver_fraction': b_driver_fraction}
    }

def process_single_step(step, base_dir, batch_size=20, I_ext=10.0):
    """Process a single step and return the results"""
    logger.info(f"Processing step {step}")

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

            # Calculate base metrics
            pre_G_reconstructed = nx.from_numpy_array(W0, create_using=nx.DiGraph)
            gm0 = global_metrics_directed(pre_G_reconstructed)

            base_syn_hist = restored.arrays['base_syn_hist']
            mean_abs_syn = np.mean(np.abs(base_syn_hist))
            base_driver_fraction = mean_abs_syn / (mean_abs_syn + I_ext)

            # Get actual batch size from data
            actual_batch_size = pruned_S_hist_batch.shape[0]
            if actual_batch_size != batch_size:
                logger.warning(f"Expected batch size {batch_size}, got {actual_batch_size}")
                batch_size = actual_batch_size

            # Process batches
            results = []
            for batch in tqdm(range(batch_size), desc=f"Processing batches for step {step}"):
                try:
                    result = process_batch(
                        batch=batch,
                        base_S_hist=base_S_hist,
                        pruned_S_hist_batch=pruned_S_hist_batch,
                        W0=W0,
                        removed_ids=removed_ids,
                        pruned_syn_hist=restored.arrays['pruned_syn_hist'],
                        n_nodes=n_nodes,
                        I_ext=I_ext
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch {batch} in step {step}: {e}")
                    continue

            # Collect results
            all_emsrs = [r['emsrs'] for r in results]
            all_lz = [r['lz'] for r in results]
            all_sp = [r['sp'] for r in results]
            all_gm = [r['gm'] for r in results]
            all_drf = [r['drf'] for r in results]

            time_end = time.time()
            logger.info(f"Step {step} processing time: {time_end - time_start:.2f} seconds")

            # Create DataFrame
            all_metrics = [all_emsrs, all_lz, all_sp, all_gm, all_drf]
            collect_rows = []

            for b in range(len(results)):
                all_metrics_batch = [all_metrics[m][b] for m in range(len(all_metrics))]
                merged = {k: v for d in all_metrics_batch for k, v in d.items()}
                merged['step'] = step  # Add step information
                collect_rows.append(merged)

            df = pd.DataFrame(collect_rows)

            # Add base repeated quantities
            gm0_renamed = {key + "_pre": value for key, value in gm0.items()}
            for col, val in gm0_renamed.items():
                df[col] = val
            df['base_driver_fraction'] = base_driver_fraction

            # Save to CSV
            df.to_csv(analysis_path, index=False)
            logger.info(f"Saved metrics for step {step} to {analysis_path}")

            return True

    except Exception as e:
        logger.error(f"Error processing step {step}: {e}")
        return False

def main():
    """Main function to process all steps"""
    logger.info(f"Found {n_of_directories} directories to process")

    successful_steps = []
    failed_steps = []

    total_start_time = time.time()

    # Process each step
    for step in range(n_of_directories):
        success = process_single_step(step, base_dir)
        if success:
            successful_steps.append(step)
        else:
            failed_steps.append(step)

    total_end_time = time.time()

    # Summary
    logger.info(f"Processing complete!")
    logger.info(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    logger.info(f"Successful steps: {successful_steps}")
    if failed_steps:
        logger.warning(f"Failed steps: {failed_steps}")

    # Optionally create a combined metrics file
    create_combined_metrics(base_dir, successful_steps)

def create_combined_metrics(base_dir, successful_steps):
    """Create a combined metrics file from all successful steps"""
    combined_data = []

    for step in successful_steps:
        metrics_path = base_dir / str(step) / 'metrics.csv'
        if metrics_path.exists():
            df_step = pd.read_csv(metrics_path)
            combined_data.append(df_step)

    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_path = base_dir / 'combined_metrics.csv'
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"Created combined metrics file: {combined_path}")

if __name__ == "__main__":
    main()
