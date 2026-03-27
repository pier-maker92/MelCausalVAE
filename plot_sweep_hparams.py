import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


RUN_DIR_RE = re.compile(r".*_n(?P<n>\d+)_t(?P<t>[-+]?\d*\.?\d+)_g(?P<g>[-+]?\d*\.?\d+)$")


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        val = float(x)
        if math.isnan(val):
            return None
        return val
    except Exception:
        return None


def parse_hparams_from_dir(run_dir: Path) -> Optional[Dict[str, float]]:
    m = RUN_DIR_RE.match(run_dir.name)
    if not m:
        return None
    return {
        "n_steps": float(m.group("n")),
        "temperature": float(m.group("t")),
        "guidance_scale": float(m.group("g")),
    }


def load_metric_rows(root_dir: Path) -> List[Dict]:
    rows: List[Dict] = []
    for metrics_path in sorted(root_dir.rglob("metrics.json")):
        run_dir = metrics_path.parent
        hparams = parse_hparams_from_dir(run_dir)
        if hparams is None:
            continue

        checkpoint_name = run_dir.parent.name
        with open(metrics_path, "r") as f:
            data = json.load(f)

        all_agg = data.get("aggregates", {}).get("all", {})
        recon = all_agg.get("reconstructed", {})
        ref = all_agg.get("ref", {})
        row = {
            "checkpoint": checkpoint_name,
            "run_dir": str(run_dir),
            **hparams,
            "recon_WER": safe_float(recon.get("WER")),
            "recon_CER": safe_float(recon.get("CER")),
            "recon_UTMOS": safe_float(recon.get("UTMOS")),
            "ref_WER": safe_float(ref.get("WER")),
            "ref_CER": safe_float(ref.get("CER")),
            "ref_UTMOS": safe_float(ref.get("UTMOS")),
        }
        rows.append(row)
    return rows


def linear_regression(rows: List[Dict], param: str, metric: str) -> Optional[Dict]:
    """Compute linear regression: metric ~ param."""
    xs = []
    ys = []
    for r in rows:
        x = r.get(param)
        y = r.get(metric)
        if x is not None and y is not None:
            xs.append(float(x))
            ys.append(float(y))
    
    if len(xs) < 2:
        return None
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)
    return {
        "param": param,
        "metric": metric,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "n_samples": len(xs),
    }


def compute_all_regressions(rows: List[Dict]) -> List[Dict]:
    """Compute linear regression for all param-metric combinations."""
    metrics = ["recon_WER", "recon_CER", "recon_UTMOS"]
    params = ["temperature", "guidance_scale", "n_steps"]
    
    results = []
    for param in params:
        for metric in metrics:
            reg = linear_regression(rows, param, metric)
            if reg is not None:
                results.append(reg)
    return results


def write_regression_summary(regressions: List[Dict], out_csv: Path):
    """Write regression coefficients to CSV."""
    cols = ["param", "metric", "slope", "intercept", "r_squared", "p_value", "n_samples"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for reg in regressions:
            vals = [str(reg.get(c, "")) for c in cols]
            f.write(",".join(vals) + "\n")


def print_regression_summary(regressions: List[Dict]):
    """Print regression summary in a readable table format."""
    print("\n" + "=" * 90)
    print("LINEAR REGRESSION SUMMARY: impact of hyperparameters on metrics")
    print("=" * 90)
    print(f"{'Param':<18} {'Metric':<15} {'Slope':<10} {'R²':<8} {'P-value':<10} {'N':<5}")
    print("-" * 90)
    for reg in regressions:
        param = reg["param"]
        metric = reg["metric"]
        slope = reg["slope"]
        r2 = reg["r_squared"]
        pval = reg["p_value"]
        n = reg["n_samples"]
        
        # Interpret impact
        if abs(slope) < 0.01:
            impact = "negligible"
        elif slope > 0:
            impact = f"+{slope:.4f}"
        else:
            impact = f"{slope:.4f}"
        
        print(f"{param:<18} {metric:<15} {impact:<10} {r2:<8.4f} {pval:<10.4g} {n:<5}")
    print("=" * 90)
    print("Interpretation: positive slope = metric increases with param (worse for WER/CER, better for UTMOS)")
    print("                negative slope = metric decreases with param (better for WER/CER, worse for UTMOS)")
    print("=" * 90 + "\n")


def aggregate_by_param(rows: List[Dict], param: str, metric: str) -> Dict[float, Dict[str, float]]:
    """Aggregate metric values by hyperparameter for plotting."""
    buckets: Dict[float, List[float]] = {}
    for r in rows:
        x = r[param]
        y = r.get(metric)
        if y is None:
            continue
        buckets.setdefault(x, []).append(y)

    out: Dict[float, Dict[str, float]] = {}
    for x, vals in buckets.items():
        out[x] = {
            "mean": mean(vals),
            "std": stdev(vals) if len(vals) > 1 else 0.0,
            "count": float(len(vals)),
        }
    return out


def plot_combined_metric_view(rows: List[Dict], out_path: Path, title: str):
    """
    One PNG with 3 subplots (temperature, guidance_scale, n_steps).
    Each subplot shows 3 curves: WER, CER, UTMOS (solid for reconstructed, dashed for reference).
    """
    recon_metrics = ["recon_WER", "recon_CER", "recon_UTMOS"]
    ref_metrics = ["ref_WER", "ref_CER", "ref_UTMOS"]
    params = ["temperature", "guidance_scale", "n_steps"]
    metric_labels = {"WER": "WER", "CER": "CER", "UTMOS": "UTMOS"}
    colors = {"WER": "C0", "CER": "C1", "UTMOS": "C2"}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, param in enumerate(params):
        ax = axes[idx]
        
        # Plot reconstructed (solid lines)
        for recon_metric in recon_metrics:
            base_name = recon_metric.replace("recon_", "")
            agg = aggregate_by_param(rows, param, recon_metric)
            xs = sorted(agg.keys())
            ys = [agg[x]["mean"] for x in xs]
            yerr = [agg[x]["std"] for x in xs]

            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    marker="o",
                    capsize=4,
                    linewidth=2,
                    linestyle="-",
                    color=colors[base_name],
                    label=f"{metric_labels[base_name]} (recon)",
                )
        
        # Plot reference (dashed lines)
        for ref_metric in ref_metrics:
            base_name = ref_metric.replace("ref_", "")
            agg = aggregate_by_param(rows, param, ref_metric)
            xs = sorted(agg.keys())
            ys = [agg[x]["mean"] for x in xs]
            yerr = [agg[x]["std"] for x in xs]

            if xs:
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    marker="s",
                    capsize=4,
                    linewidth=1.5,
                    linestyle="--",
                    color=colors[base_name],
                    alpha=0.7,
                    label=f"{metric_labels[base_name]} (ref)",
                )

        ax.set_xlabel(param.replace("_", " ").title())
        ax.set_ylabel("Metric value")
        ax.set_title(param.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_summary(rows: List[Dict], out_csv: Path):
    cols = [
        "checkpoint",
        "run_dir",
        "n_steps",
        "temperature",
        "guidance_scale",
        "recon_WER",
        "recon_CER",
        "recon_UTMOS",
        "ref_WER",
        "ref_CER",
        "ref_UTMOS",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                v = r.get(c)
                if v is None:
                    vals.append("")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot hyperparameter behavior vs WER/CER/UTMOS from eval metrics.json files.")
    parser.add_argument(
        "--metrics-root",
        type=str,
        default="/home/ec2-user/MelCausalVAE/evaluation/LibriSpeech-2ckpt-sweep-utmos-100/1d-8x-AR-VAE",
        help="Root directory that contains run folders with metrics.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/ec2-user/MelCausalVAE/evaluation/plots_hparam_behavior",
        help="Output directory for plots and CSV.",
    )
    args = parser.parse_args()

    root = Path(args.metrics_root)
    out_dir = Path(args.out_dir)

    rows = load_metric_rows(root)
    if not rows:
        raise RuntimeError(f"No parsable metrics.json found under: {root}")

    # Compute and save regression summary
    regressions = compute_all_regressions(rows)
    write_regression_summary(regressions, out_dir / "regression_summary.csv")
    print_regression_summary(regressions)

    # Global aggregated view (all checkpoints together)
    write_summary(rows, out_dir / "all_runs_summary.csv")
    plot_combined_metric_view(
        rows=rows,
        out_path=out_dir / "reconstructed_metrics_vs_hparams.png",
        title="Reconstructed metrics vs hyperparameters (all checkpoints aggregated)",
    )

    print(f"Done. Parsed {len(rows)} runs from: {root}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
