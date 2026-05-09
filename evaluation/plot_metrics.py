import os
import csv
import matplotlib.pyplot as plt
import glob
import argparse
from collections import defaultdict
import numpy as np

def plot_metrics(run_dir):
    """
    Reads all val_step_*.csv files in the run_dir, aggregates recon_UTMOS, recon_WER, 
    and recon_CER by step, and creates a progression plot.
    """
    # Find all val_step_*.csv files
    csv_files = glob.glob(os.path.join(run_dir, "val_step_*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files matching 'val_step_*.csv' found in {run_dir}")
        return

    print(f"Found {len(csv_files)} evaluation files. Processing...")

    metrics_by_step = defaultdict(lambda: {"dUTMOS": [], "dWER": [], "dCER": []})
    
    for fpath in csv_files:
        # Try to get step from filename first (val_step_10000.csv -> 10000)
        fname = os.path.basename(fpath)
        try:
            # Handle both val_step_10000.csv and similar patterns
            parts = fname.replace('.csv', '').split('_')
            step = int(parts[-1])
        except (ValueError, IndexError):
            step = None
            
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if step is None:
                        try:
                            step = int(row['step'])
                        except (KeyError, ValueError):
                            continue
                    
                    try:
                        metrics_by_step[step]["dUTMOS"].append(float(row["dUTMOS"]))
                        metrics_by_step[step]["dWER"].append(float(row["dWER"]))
                        metrics_by_step[step]["dCER"].append(float(row["dCER"]))
                    except (KeyError, ValueError):
                        continue
        except Exception as e:
            print(f"Warning: Could not read {fpath}: {e}")

    if not metrics_by_step:
        print("No valid metrics found in the CSV files.")
        return

    # Sort steps
    sorted_steps = sorted(metrics_by_step.keys())
    
    steps = []
    utmos_means = []
    wer_means = []
    cer_means = []
    
    for step in sorted_steps:
        # Only include steps that have data
        if metrics_by_step[step]["dUTMOS"]:
            steps.append(step)
            utmos_means.append(np.mean(metrics_by_step[step]["dUTMOS"]))
            wer_means.append(np.mean(metrics_by_step[step]["dWER"]))
            cer_means.append(np.mean(metrics_by_step[step]["dCER"]))

    if not steps:
        print("No data points to plot.")
        return

    # Modern plot styling
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # UTMOS on primary Y-axis
    color_utmos = '#2C3E50' # Dark blue-grey
    ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('dUTMOS (Higher is Better)', color=color_utmos, fontsize=12, fontweight='bold')
    line1 = ax1.plot(steps, utmos_means, marker='o', markersize=8, linewidth=2.5, 
                     color='#3498DB', label='dUTMOS', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_utmos)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # WER and CER on secondary Y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('dWER / dCER (Lower is Better)', color='#C0392B', fontsize=12, fontweight='bold')
    
    line2 = ax2.plot(steps, wer_means, marker='s', markersize=7, linewidth=2, 
                     color='#E74C3C', label='dWER', alpha=0.8)
    line3 = ax2.plot(steps, cer_means, marker='^', markersize=7, linewidth=2, 
                     color='#F39C12', label='dCER', alpha=0.8)
    
    ax2.tick_params(axis='y', labelcolor='#C0392B')
    ax2.set_ylim(0, max(max(wer_means), max(cer_means)) * 1.1) # Scale nicely

    # Title and Layout
    run_name = os.path.basename(run_dir.rstrip('/'))
    plt.title(f'Evaluation Metrics Progression\nRun: {run_name}', fontsize=16, pad=20)
    
    # Combined Legend
    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
               ncol=3, frameon=True, shadow=True)

    plt.tight_layout()
    
    save_path = os.path.join(run_dir, "metrics_progression.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Plot saved to: {save_path}")
    
    # Also save a copy in the evaluation root for quick access
    root_save_path = f"/scratch/piermel/MelCausalVAE/evaluation/latest_metrics_{run_name}.png"
    plt.savefig(root_save_path, dpi=300, bbox_inches='tight')
    print(f"Quick access copy saved to: {root_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot delta metrics (dUTMOS, dWER, dCER) from evaluation CSVs.")
    parser.add_argument("-i", "--run_dir", type=str, help="Directory containing val_step_*.csv files. If not provided, uses the most recent run in evaluation/validation_training.")
    
    args = parser.parse_args()
    
    target_dir = args.run_dir
    if not target_dir:
        base_eval_dir = "/scratch/piermel/MelCausalVAE/evaluation/validation_training"
        if os.path.exists(base_eval_dir):
            runs = [os.path.join(base_eval_dir, d) for d in os.listdir(base_eval_dir) if os.path.isdir(os.path.join(base_eval_dir, d))]
            if not runs:
                print(f"No run directories found in {base_eval_dir}")
            else:
                # Sort by creation time (most recent first)
                target_dir = max(runs, key=os.path.getmtime)
                print(f"No --run_dir specified. Auto-detecting most recent run: {os.path.basename(target_dir)}")
        else:
            print(f"Base directory not found: {base_eval_dir}")
    
    if target_dir:
        if os.path.isdir(target_dir):
            plot_metrics(target_dir)
        else:
            print(f"Error: {target_dir} is not a valid directory.")
