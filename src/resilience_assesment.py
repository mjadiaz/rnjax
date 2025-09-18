import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_dataframe(filepath):
    """
    Load a pandas dataframe from a file in the data_dir.
    """

    full_path = Path(filepath)

    # Determine file extension
    ext = full_path.suffix

    if ext.lower() == '.csv':
        return pd.read_csv(full_path, index_col=False)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


df = load_dataframe('save/ER_sparse_stdp/combined_metrics.csv')
print(df.columns)

# --- 0) config ---
EPS = 1e-12               # tiny guard for divisions
DEN_ABS_GUARD = 1e-8      # drop runs whose denominator is ~0
N_BOOT = 5000             # bootstrap iters for CIs (adjust or skip section if not needed)
RANDOM_STATE = 423
rng = np.random.default_rng()


den_H =  df['emsrs_h_A'] - df['emsrs_h_B']  # H(A) - H(B) -> Instant change
num_H =  df['emsrs_h_C'] - df['emsrs_h_B']  # H(C) - H(B) -> Late recovery



den_MI =  df['emsrs_mi_AA'] - df['emsrs_mi_AB']  # H(A) - I(A,B)
num_MI =  df['emsrs_mi_AC'] - df['emsrs_mi_AB']  # I(A,C) - I(A,B)



valid_den_H  = np.abs(den_H)  > DEN_ABS_GUARD
valid_den_MI = np.abs(den_MI) > DEN_ABS_GUARD


df['T_H']  = np.where(valid_den_H,  num_H / (den_H + EPS),  np.nan)
df['T_MI'] = np.where(valid_den_MI, num_MI / (den_MI + EPS), np.nan)

print(df)
# --- 3) Plotting ---
plt.figure(figsize=(10, 6))

# Sample 10 different steps from 0 to
n_steps_to_sample=   10
available_steps = df['step'].unique()
available_steps = available_steps[(available_steps >= 0) & (available_steps <= 100)]
if len(available_steps) > 10:
    sampled_steps = sorted(rng.choice(available_steps, size=n_steps_to_sample, replace=False))
else:
    sampled_steps = sorted(available_steps)

colors = plt.cm.viridis(np.linspace(0, 1, len(sampled_steps)))

print(sampled_steps)

for i, step in enumerate(sampled_steps):
    group = df[df['step'] == step]
    plt.scatter(
                group['T_MI'],
                group['T_H'],
                color=colors[i],
                label=f'Step {step}',
                alpha=0.7
    )

plt.ylabel('T_H (Entropy Transfer)')
plt.xlabel('T_MI (Mutual Information Transfer)')
plt.title('T_H vs T_MI for 10 Sampled Steps (0-100)')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.tight_layout()
#plt.savefig('transfer_metrics_by_step.png', dpi=300, bbox_inches='tight')
plt.show()
