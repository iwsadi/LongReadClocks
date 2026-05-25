"""
qc.py
~~~~~
Methylation quality-control utilities for LongReadClock.

Covers three QC analyses described in the paper:

1. **ML-score QC** — Quantifies methylation calling uncertainty using the
   ambiguous ML fraction (proportion of CpG calls with ML scores 51–204,
   i.e. probability 0.20–0.80).  Only ONT data is retained for clock
   construction based on this threshold.

2. **Bulk PCA** — Principal component analysis of genome-wide beta values
   across all sequenced participants.  Reveals platform-batch structure and
   motivates the primary ONT focus.  Faithfully reproduces the analysis in
   ``QC_plot.ipynb`` (50 000 randomly sampled CpGs, randomized PCA).

3. **ELOVL2 sanity check** — Extracts the exact beta value at the canonical
   ELOVL2 CpG (chr6:11044644) for every sample and regresses it against age.
   This is a well-known aging marker and serves as an end-to-end pipeline
   sanity check.

Extracted from ``QC_plot.ipynb`` (raw notebook).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# 1. ML-score QC metrics
# ---------------------------------------------------------------------------

def calculate_ambiguous_fraction(
    ml_scores: np.ndarray,
    lower_bound: int = 51,
    upper_bound: int = 204,
) -> float:
    """Compute the fraction of ML scores in the ambiguous range [51, 204].

    ML scores in this range correspond to methylation probabilities between
    0.20 and 0.80, where the basecaller is uncertain.  A high ambiguous
    fraction indicates poor methylation calling confidence (characteristic
    of non-ONT long-read platforms).

    Parameters
    ----------
    ml_scores : np.ndarray
        1-D array of integer ML scores (0–255) from a BAM/CRAM file.
    lower_bound, upper_bound : int
        Inclusive bounds defining the ambiguous range.

    Returns
    -------
    float
        Fraction of calls in [lower_bound, upper_bound].
    """
    if ml_scores.size == 0:
        return 0.0
    n_ambiguous = int(np.sum((ml_scores >= lower_bound) & (ml_scores <= upper_bound)))
    return float(n_ambiguous / ml_scores.size)


def calculate_mean_binary_entropy(ml_scores: np.ndarray) -> float:
    """Compute mean binary entropy across ML calls.

    H = -1/N * sum( p_i * log2(p_i) + (1-p_i) * log2(1-p_i) )
    where p_i = ML_i / 255.

    Parameters
    ----------
    ml_scores : np.ndarray
        1-D array of integer ML scores.

    Returns
    -------
    float
        Mean binary entropy (bits).  Higher → greater calling uncertainty.
    """
    if ml_scores.size == 0:
        return 0.0
    eps = 1e-12
    p   = np.clip(ml_scores.astype(np.float64) / 255.0, eps, 1.0 - eps)
    H   = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return float(np.mean(H))


# ---------------------------------------------------------------------------
# 2. Beta file loading for PCA
# ---------------------------------------------------------------------------

def load_beta_values_at_sites(
    beta_path,
    site_idx: np.ndarray,
    dtype=np.uint8,
    use_gcs: bool = False,
    gcs_client=None,
    bucket_name: Optional[str] = None,
) -> Tuple[np.ndarray, float, float]:
    """Load methylation beta fractions at specified CpG sites from a beta file.

    Parameters
    ----------
    beta_path : Path or str
        Local path or GCS path (``gs://bucket/path``) to a ``.beta`` file.
    site_idx : np.ndarray of int
        0-based CpG indices to extract.
    dtype : numpy dtype
        Stored dtype of the beta file (uint8 for wgbstools output).
    use_gcs : bool
        Set ``True`` to load from GCS.
    gcs_client, bucket_name : optional
        GCS client and bucket name.

    Returns
    -------
    beta : np.ndarray of float32, shape (len(site_idx),)
        Methylation fractions; NaN where uncovered.
    coverage_fraction : float
        Fraction of requested sites with ≥ 1 read.
    mean_depth : float
        Mean read depth across requested sites (NaN-safe).
    """
    path_str = str(beta_path)

    if use_gcs:
        if gcs_client is None:
            from google.cloud import storage as gcs
            gcs_client = gcs.Client()
        bucket = gcs_client.bucket(bucket_name)
        blob_name = path_str.replace(f"gs://{bucket_name}/", "")
        raw = bucket.blob(blob_name).download_as_bytes()
        arr = np.frombuffer(raw, dtype=dtype)
    else:
        arr = np.fromfile(path_str, dtype=dtype)

    if arr.size % 2:
        arr = arr[:-1]
    pairs = arr.reshape(-1, 2)

    if pairs.shape[0] <= int(site_idx.max()):
        raise ValueError(
            f"Beta file too short: n_sites={pairs.shape[0]}, "
            f"max_site_requested={int(site_idx.max())}"
        )

    m = pairs[site_idx, 0].astype(np.float32)
    t = pairs[site_idx, 1].astype(np.float32)

    beta = np.full(len(site_idx), np.nan, dtype=np.float32)
    cov  = t > 0
    beta[cov] = m[cov] / t[cov]

    cov_frac  = float(np.mean(cov))
    mean_dep  = float(np.nanmean(t)) if cov.any() else np.nan

    return beta, cov_frac, mean_dep


# ---------------------------------------------------------------------------
# 3. PCA pipeline
# ---------------------------------------------------------------------------

def build_pca_matrix(
    beta_paths: List,
    n_cpgs_sampled:     int = 50_000,
    n_total_cpgs:       int = 29_152_891,
    random_seed:        int = 20260426,
    min_sample_obs_frac: float = 0.50,
    min_site_obs_frac:  float = 0.80,
    use_gcs:            bool = False,
    gcs_client=None,
    bucket_name:        Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build a sample × CpG matrix for PCA.

    Randomly samples ``n_cpgs_sampled`` CpG positions genome-wide, loads
    beta values for each sample at those positions, applies missingness
    filters, and mean-centers the matrix.

    Parameters
    ----------
    beta_paths : list
        Paths or GCS paths to all ``.beta`` files.
    n_cpgs_sampled : int
        Number of CpG sites to sample (default 50 000, matching the paper).
    n_total_cpgs : int
        Total number of CpG sites in the genome index (hg38 = 29 152 891).
    random_seed : int
        Seed for reproducible site sampling (default 20260426).
    min_sample_obs_frac : float
        Minimum fraction of sampled sites a sample must have covered.
    min_site_obs_frac : float
        Minimum fraction of samples covering a site for it to be retained.
    use_gcs : bool
        Load from GCS if ``True``.

    Returns
    -------
    X_centered : np.ndarray, shape (n_samples_kept, n_sites_kept)
        Mean-centred beta matrix ready for PCA.
    site_idx_kept : np.ndarray
        Genome-wide CpG indices of the retained sites.
    sample_meta : pd.DataFrame
        One row per retained sample with ``sample_key`` and ``matrix_status``.
    """
    rng      = np.random.default_rng(random_seed)
    site_idx = np.sort(
        rng.choice(n_total_cpgs, size=n_cpgs_sampled, replace=False)
    ).astype(np.int32)

    n        = len(beta_paths)
    n_sites  = len(site_idx)
    X        = np.full((n, n_sites), np.nan, dtype=np.float32)
    rows     = []

    for i, path in enumerate(beta_paths):
        sample_key = Path(str(path)).stem
        try:
            vals, cov_frac, mean_dep = load_beta_values_at_sites(
                path, site_idx,
                use_gcs=use_gcs,
                gcs_client=gcs_client,
                bucket_name=bucket_name,
            )
            X[i, :] = vals
            rows.append({
                "sample_key":    sample_key,
                "matrix_status": "ok",
                "cov_frac_50k":  cov_frac,
                "mean_depth_50k": mean_dep,
            })
        except Exception as exc:
            rows.append({
                "sample_key":    sample_key,
                "matrix_status": f"error:{type(exc).__name__}:{str(exc)[:120]}",
                "cov_frac_50k":  np.nan,
                "mean_depth_50k": np.nan,
            })

    meta_df = pd.DataFrame(rows)

    # Filter failed rows
    ok_mask = meta_df["matrix_status"].eq("ok").values
    X_ok    = X[ok_mask]
    meta_ok = meta_df[ok_mask].reset_index(drop=True)

    # Sample missingness filter
    sample_obs = np.mean(~np.isnan(X_ok), axis=1)
    keep_samp  = sample_obs >= min_sample_obs_frac
    X2         = X_ok[keep_samp]
    meta2      = meta_ok[keep_samp].reset_index(drop=True)
    meta2["sample_obs_frac_50k"] = sample_obs[keep_samp]

    # Site missingness filter
    site_obs    = np.mean(~np.isnan(X2), axis=0)
    keep_sites  = site_obs >= min_site_obs_frac
    X3          = X2[:, keep_sites]
    site_idx_kept = site_idx[keep_sites]

    # Mean imputation + centering
    col_means = np.nanmean(X3, axis=0).astype(np.float32)
    nan_mask  = np.isnan(X3)
    X3[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    X_centered   = (X3 - col_means).astype(np.float32)

    print(
        f"[qc] PCA matrix: {X_centered.shape[0]} samples × "
        f"{X_centered.shape[1]} sites "
        f"(from {n} input, {int(keep_samp.sum())} after sample filter)"
    )
    return X_centered, site_idx_kept, meta2


def run_pca(
    X_centered: np.ndarray,
    n_components: int = 2,
    random_seed: int = 20260426,
) -> Tuple[PCA, np.ndarray]:
    """Fit randomized PCA on a mean-centred beta matrix.

    Parameters
    ----------
    X_centered : np.ndarray
        Output of :func:`build_pca_matrix`.
    n_components : int
        Number of PCs to compute (default 2).
    random_seed : int
        Random state for reproducibility.

    Returns
    -------
    pca : sklearn PCA object
    scores : np.ndarray, shape (n_samples, n_components)
    """
    pca    = PCA(n_components=n_components, svd_solver="randomized",
                 random_state=random_seed)
    scores = pca.fit_transform(X_centered)
    print(
        f"[qc] PC1 var={100*pca.explained_variance_ratio_[0]:.2f}%  "
        f"PC2 var={100*pca.explained_variance_ratio_[1]:.2f}%"
    )
    return pca, scores


# ---------------------------------------------------------------------------
# 4. ELOVL2 sanity check
# ---------------------------------------------------------------------------

# hg38 wgbstools CpG index for chr6:11044644
# Verified in QC_plot.ipynb: start0 == 11044644, cpg_index == 9398401
ELOVL2_CPG_INDEX = 9398401
ELOVL2_CHROM     = "chr6"
ELOVL2_POS_1BASED = 11044644


def extract_elovl2_beta(
    beta_path,
    cpg_index: int = ELOVL2_CPG_INDEX,
    dtype=np.uint8,
    use_gcs: bool = False,
    gcs_client=None,
    bucket_name: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Extract methylated count, total depth, and beta at the ELOVL2 CpG.

    Parameters
    ----------
    beta_path : Path or str
        Path to a ``.beta`` file.
    cpg_index : int
        0-based wgbstools CpG index for the target locus.
        Default is the verified ELOVL2 index for hg38.

    Returns
    -------
    (m, t, beta) : (float, float, float)
        Methylated count, total depth, and beta fraction (NaN if uncovered).
    """
    vals, _, _ = load_beta_values_at_sites(
        beta_path,
        np.array([cpg_index], dtype=np.int32),
        dtype=dtype,
        use_gcs=use_gcs,
        gcs_client=gcs_client,
        bucket_name=bucket_name,
    )
    beta_val = float(vals[0])

    # Also retrieve raw counts
    path_str = str(beta_path)
    if use_gcs:
        if gcs_client is None:
            from google.cloud import storage as gcs
            gcs_client = gcs.Client()
        bucket = gcs_client.bucket(bucket_name)
        raw  = bucket.blob(path_str.replace(f"gs://{bucket_name}/", "")).download_as_bytes()
        arr  = np.frombuffer(raw, dtype=dtype)
    else:
        arr = np.fromfile(path_str, dtype=dtype)

    if arr.size % 2:
        arr = arr[:-1]
    pairs = arr.reshape(-1, 2)
    m = float(pairs[cpg_index, 0]) if cpg_index < pairs.shape[0] else np.nan
    t = float(pairs[cpg_index, 1]) if cpg_index < pairs.shape[0] else np.nan

    return m, t, beta_val


def run_elovl2_validation(
    beta_paths:  List,
    sample_ages: Dict[str, float],
    cpg_index:   int = ELOVL2_CPG_INDEX,
    use_gcs:     bool = False,
    gcs_client=None,
    bucket_name: Optional[str] = None,
) -> pd.DataFrame:
    """Extract ELOVL2 beta values across all samples for age-regression validation.

    Parameters
    ----------
    beta_paths : list of Path or str
        Beta files to process.
    sample_ages : dict
        Mapping sample_id → chronological age.
    cpg_index : int
        wgbstools CpG index (default = verified ELOVL2 for hg38).

    Returns
    -------
    pd.DataFrame
        Columns: ``sample_key``, ``age``, ``elovl2_m``, ``elovl2_t``,
        ``elovl2_beta``, ``status``.
    """
    rows = []
    for path in beta_paths:
        sample_key = Path(str(path)).stem
        age        = sample_ages.get(sample_key, np.nan)
        try:
            m, t, beta = extract_elovl2_beta(
                path, cpg_index,
                use_gcs=use_gcs,
                gcs_client=gcs_client,
                bucket_name=bucket_name,
            )
            status = "ok"
        except Exception as exc:
            m, t, beta = np.nan, np.nan, np.nan
            status = f"error:{type(exc).__name__}:{str(exc)[:80]}"

        rows.append({
            "sample_key":  sample_key,
            "age":         age,
            "elovl2_m":    m,
            "elovl2_t":    t,
            "elovl2_beta": beta,
            "status":      status,
        })

    return pd.DataFrame(rows)
