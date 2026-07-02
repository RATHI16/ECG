"""Statistical distribution fitting and sampling for ECG parameters."""

import numpy as np
from scipy import stats


def fit_best_distribution(data, candidates=None):
    """Fit data to candidate distributions, return best by KS test.

    Returns: (dist_name, params, ks_pvalue)
    """
    if candidates is None:
        candidates = ["norm", "lognorm", "gamma"]

    data = np.array(data)
    data = data[~np.isnan(data)]
    if len(data) < 5:
        return "norm", stats.norm.fit(data), 0.0

    best_name = "norm"
    best_p = -1
    best_params = stats.norm.fit(data)

    for name in candidates:
        try:
            dist = getattr(stats, name)
            # For lognorm and gamma, data must be positive
            if name in ("lognorm", "gamma") and np.any(data <= 0):
                continue
            params = dist.fit(data)
            D, p = stats.kstest(data, name, args=params)
            if p > best_p:
                best_p = p
                best_name = name
                best_params = params
        except Exception:
            continue

    return best_name, best_params, best_p


def sample_from_distribution(dist_name, params, n_samples, clip_range=None):
    """Sample n values from a fitted distribution.

    Args:
        dist_name: scipy.stats distribution name
        params: distribution parameters from fit()
        n_samples: number of samples to generate
        clip_range: optional (min, max) tuple to clip values
    """
    dist = getattr(stats, dist_name)
    samples = dist.rvs(*params, size=n_samples)
    if clip_range is not None:
        samples = np.clip(samples, clip_range[0], clip_range[1])
    return samples


def compute_correlation_matrix(feature_df):
    """Compute Pearson correlation matrix from feature DataFrame."""
    return feature_df.corr().fillna(0)


def generate_correlated_samples(means, stds, corr_matrix, n_samples):
    """Generate correlated samples using Cholesky decomposition.

    Args:
        means: array of feature means
        stds: array of feature standard deviations
        corr_matrix: correlation matrix (n_features x n_features)
        n_samples: number of samples to generate

    Returns: array of shape (n_samples, n_features)
    """
    n_features = len(means)
    # Ensure correlation matrix is positive definite
    eigvals = np.linalg.eigvalsh(corr_matrix)
    if np.any(eigvals <= 0):
        # Add small diagonal perturbation
        corr_matrix = corr_matrix + np.eye(n_features) * (abs(eigvals.min()) + 0.01)

    L = np.linalg.cholesky(corr_matrix)
    z = np.random.randn(n_samples, n_features)
    correlated = z @ L.T

    # Scale from standard normal to target means/stds
    samples = correlated * stds + means
    return samples
