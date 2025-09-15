"""
# Print experiment information only
python src/example_load.py --save_dir /path/to/saved/data --info_only

# Visualize data from a specific step
python src/example_load.py --save_dir /path/to/saved/data --step 2

# Visualize data from the first available step
python src/example_load.py --save_dir /path/to/saved/data
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from load_utils import (
    list_available_steps,
    load_metadata,
    load_array,
    load_step_data,
    print_experiment_info
)

def parse_args():
    parser = argparse.ArgumentParser(description='Load and visualize attack data')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Directory where attack results are saved')
    parser.add_argument('--step', type=int, default=None,
                      help='Specific step to visualize (default: first available step)')
    parser.add_argument('--info_only', action='store_true',
                      help='Only print experiment info without visualization')
    return parser.parse_args()

def visualize_step_data(data, step):
    """Visualize key data from a step."""
    print(f"\nVisualizing data from step {step}")
    print("-" * 50)

    # Extract metadata
    metadata = data['metadata']
    print(f"Experiment type: {metadata['experiment_type']}")
    print(f"Parameters: {metadata['parameters']}")

    # Get arrays
    arrays = data['arrays']

    # Plot spike histograms
    if 'base_S_hist' in arrays and 'pruned_S_hist_batch' in arrays:
        base_S_hist = arrays['base_S_hist']
        pruned_S_hist_batch = arrays['pruned_S_hist_batch']

        # Plotting
        plt.figure(figsize=(15, 10))

        # Base network spikes
        plt.subplot(2, 1, 1)
        plt.title(f"Base Network Spike History (N={metadata['parameters']['N']})")
        plt.imshow(base_S_hist.T, aspect='auto', cmap='binary')
        plt.xlabel('Time Step')
        plt.ylabel('Neuron Index')

        # Pruned network spikes (first batch example)
        plt.subplot(2, 1, 2)
        plt.title(f"Pruned Network Spike History (Example 0, removed {metadata['parameters']['attack_fraction']*100}% neurons)")
        plt.imshow(pruned_S_hist_batch[0].T, aspect='auto', cmap='binary')
        plt.xlabel('Time Step')
        plt.ylabel('Neuron Index')

        plt.tight_layout()

        # Create output directory
        output_dir = Path('visualizations')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f'spike_history_step_{step}.png')
        print(f"Visualization saved to visualizations/spike_history_step_{step}.png")

        # Driver fraction analysis
        if 'base_driver_fraction' in arrays and 'batch_driver_fraction' in arrays:
            base_df = arrays['base_driver_fraction']
            print(base_df)
            print(type(base_df))
            batch_df = arrays['batch_driver_fraction']
            print(batch_df)
            print(type(batch_df))

            print(f"Base network driver fraction: {base_df[0]:.4f}")
            print(f"Pruned networks mean driver fraction: {np.mean(batch_df):.4f}")
            print(f"Pruned networks driver fraction std: {np.std(batch_df):.4f}")

            # Plot driver fraction distribution
            plt.figure(figsize=(10, 6))
            plt.hist(batch_df, bins=20, alpha=0.7)
            plt.axvline(base_df[0], color='r', linestyle='dashed', linewidth=2, label=f'Base: {base_df[0]:.4f}')
            plt.axvline(np.mean(batch_df), color='g', linestyle='dashed', linewidth=2,
                        label=f'Mean: {np.mean(batch_df):.4f}')
            plt.title('Driver Fraction Distribution after Pruning')
            plt.xlabel('Driver Fraction')
            plt.ylabel('Count')
            plt.legend()
            plt.savefig(output_dir / f'driver_fraction_step_{step}.png')
            print(f"Driver fraction plot saved to visualizations/driver_fraction_step_{step}.png")
    else:
        print("Required arrays not found for visualization")

def main():
    args = parse_args()
    save_dir = Path(args.save_dir)

    # Print experiment information
    print_experiment_info(save_dir)

    if args.info_only:
        return

    # Get steps
    steps = list_available_steps(save_dir)
    if not steps:
        print("No steps found in the save directory")
        return

    # Determine which step to visualize
    step = args.step if args.step is not None else steps[0]
    if step not in steps:
        print(f"Step {step} not found. Available steps: {steps}")
        return

    # Load step data
    data = load_step_data(save_dir, step)

    # Visualize step data
    visualize_step_data(data, step)

    print("\nDone!")

if __name__ == "__main__":
    main()
