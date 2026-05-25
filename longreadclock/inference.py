"""
inference.py
~~~~~~~~~~~~
Bernoulli grid-search age inference for LongReadClock.

Age is inferred by finding the age on a discrete grid (0–100 years, step 0.1)
that maximises the Bernoulli log-likelihood across the top-K age-informative
CpG sites.  This strategy, adapted from scAge and LongReadAge, is robust to
sparse per-sample coverage because it uses only the sites that are actually
observed in a given sample.

Extracted from ``age_reference.ipynb`` and ``cross_reference_ont_0_100.ipynb``
(raw notebooks).

Key constants (matching the published analysis):
    K_INFER = 1000   # top age-informative sites ranked by |t_stat|
    CAP = 20.0       # depth cap (prevents high-coverage sites from dominating)
    EPS = 1e-3       # probability clipping for log-likelihood stability
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .reference import (
    construct_reference_from_betas,
    select_eligible_sites,
    _load_beta_pairs_local,
)


# ---------------------------------------------------------------------------
# Single-sample age prediction
# ---------------------------------------------------------------------------

def predict_age_bernoulli_grid(
    beta_path: Path,
    ref_df: pd.DataFrame,
    eligible_row_idx: np.ndarray,
    top_k: int = 1000,
    age_min: float = 0.0,
    age_max: float = 100.0,
    age_step: float = 0.1,
    depth_cap: float = 20.0,
    eps: float = 1e-3,
    use_gcs: bool = False,
    bucket_name: Optional[str] = None,
    gcs_client=None,
) -> float:
    """Predict biological age from a .beta file using Bernoulli grid search.

    The algorithm:
    1. Load methylation counts from the beta file.
    2. Restrict to the eligible sites (ranked by |t_stat|).
    3. Further filter to the top-K sites that are actually covered.
    4. Apply a depth cap to down-weight extreme coverage.
    5. Evaluate Bernoulli log-likelihood at each age on the grid.
    6. Return the age with the maximum log-likelihood.

    Parameters
    ----------
    beta_path : Path or str
        Path to the ``.beta`` file (local or GCS).
    ref_df : pd.DataFrame
        Reference DataFrame from :func:`~longreadclock.reference.construct_reference_from_betas`.
    eligible_row_idx : np.ndarray
        Row indices into ``ref_df`` of eligible sites, sorted by |t_stat|
        descending (from :func:`~longreadclock.reference.select_eligible_sites`).
    top_k : int
        Number of top sites to use (default 1 000).
    age_min, age_max, age_step : float
        Discrete age grid parameters.
    depth_cap : float
        Maximum read depth used per site (caps extreme coverage).
    eps : float
        Probability clipping to avoid log(0).
    use_gcs : bool
        If ``True``, load the beta file from GCS.
    bucket_name : str, optional
        GCS bucket name (required when ``use_gcs=True``).
    gcs_client : optional
        ``google.cloud.storage.Client`` instance (created if not provided).

    Returns
    -------
    float
        Predicted biological age in years.
    """
    # 1. Load beta file
    if use_gcs:
        if gcs_client is None:
            from google.cloud import storage as gcs
            gcs_client = gcs.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob_name = str(beta_path).replace(f"gs://{bucket_name}/", "")
        blob = bucket.blob(blob_name)
        from .reference import _load_beta_pairs_gcs
        pairs = _load_beta_pairs_gcs(blob)
        if pairs is None:
            return 50.0
        meth  = pairs[:, 0].astype(np.float64)
        total = pairs[:, 1].astype(np.float64)
    else:
        pairs = _load_beta_pairs_local(Path(beta_path))
        meth  = pairs[:, 0].astype(np.float64)
        total = pairs[:, 1].astype(np.float64)

    # 2. Extract eligible sites
    # eligible_row_idx are row positions in ref_df; map to genome site_index
    ref_eligible = ref_df.iloc[eligible_row_idx].reset_index(drop=True)
    site_indices = ref_eligible["site_index"].values.astype(np.int64)

    # Guard against beta files shorter than expected
    valid_in_beta = site_indices < len(meth)
    if not valid_in_beta.any():
        return 50.0

    m_el = meth[site_indices[valid_in_beta]]
    t_el = total[site_indices[valid_in_beta]]
    ref_el = ref_eligible.iloc[valid_in_beta].reset_index(drop=True)

    # 3. Filter to covered sites
    cov_mask = t_el > 0
    if not cov_mask.any():
        return 50.0

    m_cov   = m_el[cov_mask]
    t_cov   = t_el[cov_mask]
    ref_cov = ref_el.iloc[cov_mask].reset_index(drop=True)

    # 4. Select top-K by |t_stat|
    abs_t    = np.abs(ref_cov["t_stat"].values)
    sort_idx = np.argsort(-abs_t)[:top_k]

    m_sel      = m_cov[sort_idx]
    t_sel      = t_cov[sort_idx]
    slopes     = ref_cov["slope"].values[sort_idx]
    intercepts = ref_cov["intercept"].values[sort_idx]

    # 5. Apply depth cap
    t_cap    = np.minimum(t_sel, depth_cap)
    scale    = np.where(t_sel > 0, t_cap / t_sel, 0.0)
    m_cap    = m_sel * scale

    # 6. Bernoulli grid log-likelihood
    age_grid = np.arange(age_min, age_max + age_step / 2, age_step)
    log_lik  = np.zeros(len(age_grid), dtype=np.float64)

    for idx, age in enumerate(age_grid):
        p = np.clip(intercepts + slopes * age, eps, 1.0 - eps)
        log_lik[idx] = np.sum(
            m_cap * np.log(p) + (t_cap - m_cap) * np.log1p(-p)
        )

    return float(age_grid[np.argmax(log_lik)])


# ---------------------------------------------------------------------------
# 5-fold cross-validation
# ---------------------------------------------------------------------------

def run_5fold_cross_validation(
    beta_paths: List[Path],
    sample_ages: Dict[str, float],
    min_samples_per_site: int = 20,
    top_k: int = 1000,
    random_seed: int = 20260414,
    use_gcs: bool = False,
    bucket_name: Optional[str] = None,
) -> pd.DataFrame:
    """Run 5-fold cross-validation on a cohort of beta files.

    In each fold:
    1. Train a reference clock on the 4 training folds.
    2. Select eligible sites from the training reference.
    3. Predict age on the held-out fold using the Bernoulli grid search.

    Parameters
    ----------
    beta_paths : list of Path
        Paths to all ``.beta`` files in the cohort.
    sample_ages : dict
        Mapping sample_id → true age.
    min_samples_per_site : int
        Passed to :func:`~longreadclock.reference.construct_reference_from_betas`.
    top_k : int
        Top-K sites for age inference.
    random_seed : int
        Random seed for fold splitting (default matches the notebook).
    use_gcs : bool
        Pass ``True`` if beta files are on GCS.
    bucket_name : str, optional
        GCS bucket name (required when ``use_gcs=True``).

    Returns
    -------
    pd.DataFrame
        Columns: ``sample_id``, ``fold``, ``true_age``, ``predicted_age``,
        ``ead_residual`` (predicted − true).
    """
    from sklearn.model_selection import KFold

    samples = sorted(sample_ages.keys())
    kf      = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    # Build sample_id → path mapping
    path_map: Dict[str, Path] = {}
    for p in beta_paths:
        stem = Path(str(p)).stem
        path_map[stem] = p

    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(samples)):
        print(f"\n--- Fold {fold + 1}/5 ---")
        train_samples = [samples[i] for i in train_idx]
        test_samples  = [samples[i] for i in test_idx]

        train_paths = [path_map[s] for s in train_samples if s in path_map]

        # Build reference on training fold
        ref_df = construct_reference_from_betas(
            train_paths, sample_ages,
            min_samples_per_site=min_samples_per_site,
            use_gcs=use_gcs,
            bucket_name=bucket_name,
        )

        # Select eligible sites
        eligible = select_eligible_sites(ref_df)

        # Predict on held-out test samples
        for s in test_samples:
            if s not in path_map:
                continue
            true_age = sample_ages[s]
            pred_age = predict_age_bernoulli_grid(
                path_map[s], ref_df, eligible,
                top_k=top_k,
                use_gcs=use_gcs,
                bucket_name=bucket_name,
            )
            results.append({
                "sample_id":      s,
                "fold":           fold + 1,
                "true_age":       true_age,
                "predicted_age":  pred_age,
                "ead_residual":   pred_age - true_age,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Cross-platform / cross-reference benchmarking
# ---------------------------------------------------------------------------

def run_cross_platform_benchmarks(
    reference_models: Dict[str, Tuple[pd.DataFrame, np.ndarray]],
    test_datasets:    Dict[str, List[Path]],
    sample_ages:      Dict[str, float],
    top_k:            int = 1000,
    use_gcs:          bool = False,
    bucket_name:      Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate multiple reference models against multiple test sets.

    Reproduces the cross-platform evaluation in
    ``cross_reference_ont_0_100.ipynb``.

    Parameters
    ----------
    reference_models : dict
        Mapping platform/group name → (ref_df, eligible_row_idx).
    test_datasets : dict
        Mapping platform/group name → list of beta file paths.
    sample_ages : dict
        Mapping sample key → true age.
    top_k : int
        Top-K sites for inference.

    Returns
    -------
    pd.DataFrame
        Long-form results with columns ``reference_platform``,
        ``test_platform``, ``sample_id``, ``true_age``,
        ``predicted_age``, ``ead_residual``.
    """
    results = []

    for ref_name, (ref_df, eligible) in reference_models.items():
        for test_name, paths in test_datasets.items():
            print(f"Reference: {ref_name}  →  Test: {test_name}")
            for path in paths:
                sample_key = Path(str(path)).stem
                age_val    = sample_ages.get(sample_key)
                if age_val is None:
                    # Fuzzy match
                    for k, v in sample_ages.items():
                        if k in sample_key or sample_key in k:
                            age_val = v
                            break
                if age_val is None:
                    continue

                try:
                    pred = predict_age_bernoulli_grid(
                        path, ref_df, eligible,
                        top_k=top_k,
                        use_gcs=use_gcs,
                        bucket_name=bucket_name,
                    )
                    results.append({
                        "reference_platform": ref_name,
                        "test_platform":      test_name,
                        "sample_id":          sample_key,
                        "true_age":           age_val,
                        "predicted_age":      pred,
                        "ead_residual":       pred - age_val,
                    })
                except Exception as exc:
                    print(f"  Warning: failed for {sample_key}: {exc}")

    return pd.DataFrame(results)
