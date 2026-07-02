"""Step 4: Fit statistical distributions to extracted beat parameters.

For each numeric feature in the parameter database:
- Fit candidate distributions (normal, lognormal, gamma)
- Select best fit using KS test
- Compute correlation matrix between parameters
- Save distribution summary and correlation matrix
"""

import numpy as np
import pandas as pd
import yaml
from stats_fitting import fit_best_distribution, compute_correlation_matrix


def fit_all_distributions(parameter_db, config):
    """Fit distributions to all numeric features in the parameter database.

    Args:
        parameter_db: DataFrame from feature extraction step
        config: configuration dict

    Returns:
        dist_summary: DataFrame with fitted distribution info per feature
        corr_matrix: correlation matrix DataFrame
        fit_results: dict mapping feature -> (dist_name, params) for sampling
    """
    print("\n" + "=" * 60)
    print("STEP 4: Fitting Statistical Distributions")
    print("=" * 60)

    numeric_cols = ['rr_ms', 'hr_bpm', 'qrs_width_ms', 'r_amplitude', 'beat_length_ms']
    fit_results = {}
    summary_rows = []

    for col in numeric_cols:
        data = parameter_db[col].dropna().values
        if len(data) < 5:
            print(f"  SKIP: {col} - insufficient data ({len(data)} values)")
            continue

        dist_name, params, pval = fit_best_distribution(data)
        fit_results[col] = (dist_name, params)

        # Compute descriptive stats
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)

        summary_rows.append({
            'feature': col,
            'best_dist': dist_name,
            'params': str(params),
            'ks_pvalue': pval,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'n_samples': len(data),
        })

        print(f"  {col:20s}: {dist_name:10s} (p={pval:.4f}) "
              f"mean={mean_val:.1f} std={std_val:.1f} [{min_val:.1f}, {max_val:.1f}]")

    dist_summary = pd.DataFrame(summary_rows)

    # Correlation matrix
    corr_cols = [c for c in numeric_cols if c in parameter_db.columns]
    corr_matrix = parameter_db[corr_cols].corr().fillna(0)

    print(f"\n  Correlation matrix ({len(corr_cols)}x{len(corr_cols)}):")
    print(corr_matrix.round(2).to_string(index=True))
    print("=" * 60)

    return dist_summary, corr_matrix, fit_results


if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    print("Run via run_pipeline.py for full pipeline execution.")
