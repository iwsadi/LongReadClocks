"""
reference.py
~~~~~~~~~~~~
Age-reference model construction for LongReadClock.

The reference is built by performing CpG-site-level ordinary-least-squares
regression of methylation beta values against chronological age across a
training cohort.  Because the beta files for ~2,800 participants collectively
represent tens of GB of data, the regression is computed incrementally using
sufficient statistics (N, Sx, Sy, Sxx, Syy, Sxy) so that the entire cohort
can be processed without loading all samples into memory at once.

Extracted from ``age_reference.ipynb`` (raw notebook), which runs on the
All of Us Workbench.  The GCS I/O logic (google-cloud-storage) is used when
``use_gcs=True``; for local testing you can pass a list of local .beta file
paths with ``use_gcs=False``.

Key constants (from the notebook):
    MIN_SAMPLES_PER_SITE = 20   # minimum observations to include a site
    ABSR_MIN = 0.30             # |R| threshold for age-informative sites
    RANGE_MIN = 0.05            # minimum beta range across the age axis
    K_INFER = 1000              # top-K sites used for age inference
    AGE_STEP = 0.1              # discrete age grid resolution (years)
    EPS = 1e-3                  # clipping epsilon for Bernoulli log-likelihood
    CAP = 20.0                  # depth cap (read coverage ceiling)
"""

import io
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Beta file I/O
# ---------------------------------------------------------------------------

def _load_beta_pairs_local(beta_path: Path) -> np.ndarray:
    """Load a local wgbstools .beta file as a (n_sites, 2) uint8 array.

    The wgbstools beta format is a binary file of paired uint8 values:
    ``(methylated_count, total_count)`` for each CpG site in hg38 order.
    """
    data = np.fromfile(str(beta_path), dtype=np.uint8)
    if data.size % 2:
        data = data[:-1]
    return data.reshape(-1, 2)


def _load_beta_pairs_gcs(blob) -> Optional[np.ndarray]:
    """Load a GCS blob containing a .beta file as a (n_sites, 2) uint8 array.

    Parameters
    ----------
    blob : google.cloud.storage.Blob
        An already-resolved GCS blob object.
    """
    raw = blob.download_as_bytes()
    if raw is None or len(raw) < 2:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size % 2:
        arr = arr[:-1]
    pairs = arr.reshape(-1, 2)
    if pairs.shape[0] == 0:
        return None
    return pairs


# ---------------------------------------------------------------------------
# Incremental OLS regression
# ---------------------------------------------------------------------------

def construct_reference_from_betas(
    beta_paths: List[Path],
    sample_ages: Dict[str, float],
    min_samples_per_site: int = 20,
    use_gcs: bool = False,
    bucket_name: Optional[str] = None,
) -> pd.DataFrame:
    """Build a per-CpG OLS age-reference model from a collection of beta files.

    The regression is computed incrementally via sufficient statistics so that
    only one beta file (~57 MB for hg38) needs to be in memory at a time.

    Parameters
    ----------
    beta_paths : list of Path
        Paths to ``.beta`` files (local paths or GCS paths as strings when
        ``use_gcs=True``).
    sample_ages : dict
        Mapping of sample identifier → chronological age (years).  The sample
        identifier is matched against the stem of each beta file path
        (i.e. ``Path(p).stem``).
    min_samples_per_site : int
        Minimum number of covered observations required at a CpG site before
        it is included in the reference.  Sites with fewer observations are
        excluded.  Default 20 (matches the notebook).
    use_gcs : bool
        If ``True``, ``beta_paths`` are treated as ``gs://bucket/path``
        strings and loaded via ``google-cloud-storage``.
    bucket_name : str, optional
        GCS bucket name (required when ``use_gcs=True``).

    Returns
    -------
    pd.DataFrame
        One row per eligible CpG site with columns:
        ``site_index``, ``N``, ``slope``, ``intercept``, ``R``,
        ``t_stat``, ``P_value``.
        ``site_index`` is the 0-based genome-wide CpG index.
    """
    if use_gcs:
        from google.cloud import storage as gcs
        gcs_client = gcs.Client()
        bucket = gcs_client.bucket(bucket_name)

    # Sufficient statistics accumulators — initialised lazily on first sample
    N: Optional[np.ndarray] = None
    Sx: Optional[np.ndarray] = None
    Sy: Optional[np.ndarray] = None
    Sxx: Optional[np.ndarray] = None
    Syy: Optional[np.ndarray] = None
    Sxy: Optional[np.ndarray] = None
    expected_sites: Optional[int] = None

    processed = 0
    skipped_empty = 0
    skipped_mismatch = 0
    skipped_no_age = 0
    skipped_error = 0

    for path in beta_paths:
        path_str = str(path)
        sample_key = Path(path_str).stem

        age = sample_ages.get(sample_key)
        if age is None:
            # Try partial matching
            for k, v in sample_ages.items():
                if k in sample_key or sample_key in k:
                    age = v
                    break
        if age is None:
            skipped_no_age += 1
            continue

        try:
            if use_gcs:
                blob_name = path_str.replace(f"gs://{bucket_name}/", "")
                blob = bucket.blob(blob_name)
                pairs = _load_beta_pairs_gcs(blob)
            else:
                pairs = _load_beta_pairs_local(Path(path_str))

            if pairs is None or pairs.shape[0] == 0:
                skipped_empty += 1
                continue

            if expected_sites is None:
                expected_sites = pairs.shape[0]
                N   = np.zeros(expected_sites, dtype=np.float32)
                Sx  = np.zeros(expected_sites, dtype=np.float32)
                Sy  = np.zeros(expected_sites, dtype=np.float32)
                Sxx = np.zeros(expected_sites, dtype=np.float32)
                Syy = np.zeros(expected_sites, dtype=np.float32)
                Sxy = np.zeros(expected_sites, dtype=np.float32)

            if pairs.shape[0] != expected_sites:
                skipped_mismatch += 1
                continue

            meth  = pairs[:, 0].astype(np.float32)
            total = pairs[:, 1].astype(np.float32)

            cov_mask = total > 0
            if not np.any(cov_mask):
                skipped_empty += 1
                continue

            beta = np.zeros(expected_sites, dtype=np.float32)
            np.divide(meth, total, out=beta, where=cov_mask)

            a = float(age)
            N[cov_mask]   += 1
            Sx[cov_mask]  += a
            Sy[cov_mask]  += beta[cov_mask]
            Sxx[cov_mask] += a * a
            Syy[cov_mask] += beta[cov_mask] ** 2
            Sxy[cov_mask] += a * beta[cov_mask]

            processed += 1

        except Exception as exc:
            skipped_error += 1
            print(f"[reference] failed on {sample_key}: {exc}")

    print(
        f"[reference] processed={processed}, "
        f"skipped_no_age={skipped_no_age}, "
        f"skipped_empty={skipped_empty}, "
        f"skipped_mismatch={skipped_mismatch}, "
        f"skipped_error={skipped_error}"
    )

    if processed == 0:
        raise ValueError("No valid samples were processed. Check beta_paths and sample_ages.")

    # Sites with sufficient observations
    valid_mask = N >= min_samples_per_site
    print(f"[reference] sites with N >= {min_samples_per_site}: {valid_mask.sum():,} / {len(N):,}")

    vN   = N[valid_mask]
    vSx  = Sx[valid_mask]
    vSy  = Sy[valid_mask]
    vSxx = Sxx[valid_mask]
    vSyy = Syy[valid_mask]
    vSxy = Sxy[valid_mask]

    numerator   = (vN * vSxy) - (vSx * vSy)
    denominator = (vN * vSxx) - (vSx ** 2)

    slope     = np.divide(numerator, denominator,
                          out=np.zeros_like(numerator), where=denominator != 0)
    intercept = (vSy - slope * vSx) / vN

    denom_y = (vN * vSyy) - (vSy ** 2)
    r_denom = np.sqrt(np.maximum(denominator * denom_y, 0.0))
    R = np.divide(numerator, r_denom,
                  out=np.zeros_like(numerator), where=r_denom > 0)

    dfree  = vN - 2
    r_safe = np.clip(R, -0.999999, 0.999999)
    t_stat = r_safe * np.sqrt(dfree / (1 - r_safe ** 2))
    P_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), dfree))

    ref_df = pd.DataFrame({
        "site_index": np.where(valid_mask)[0].astype(np.int32),
        "N":          vN.astype(np.float32),
        "slope":      slope.astype(np.float32),
        "intercept":  intercept.astype(np.float32),
        "R":          R.astype(np.float32),
        "t_stat":     t_stat.astype(np.float32),
        "P_value":    P_value.astype(np.float64),
    })

    return ref_df


# ---------------------------------------------------------------------------
# Site eligibility filtering
# ---------------------------------------------------------------------------

def select_eligible_sites(
    ref_df: pd.DataFrame,
    absr_min: float = 0.30,
    range_min: float = 0.05,
    age_range: Tuple[float, float] = (18.0, 90.0),
) -> np.ndarray:
    """Select age-informative CpG sites from a reference DataFrame.

    Applies the same filters used in the notebook:
    - |R| >= ``absr_min``  (minimum correlation with age)
    - predicted beta range across ``age_range`` >= ``range_min``
      (site must vary meaningfully across the cohort age span)

    Parameters
    ----------
    ref_df : pd.DataFrame
        Output of :func:`construct_reference_from_betas`.
    absr_min : float
        Minimum |R| for a site to be considered age-informative (default 0.30).
    range_min : float
        Minimum predicted methylation range across the age span (default 0.05).
    age_range : (float, float)
        (min_age, max_age) used to evaluate the predicted beta range.

    Returns
    -------
    np.ndarray of int
        Integer positions (row indices into ``ref_df``) of eligible sites,
        sorted by |t_stat| descending.
    """
    age_lo, age_hi = age_range

    abs_r  = np.abs(ref_df["R"].values)
    beta_lo = ref_df["intercept"].values + ref_df["slope"].values * age_lo
    beta_hi = ref_df["intercept"].values + ref_df["slope"].values * age_hi
    beta_range = np.abs(beta_hi - beta_lo)

    eligible_mask = (abs_r >= absr_min) & (beta_range >= range_min)
    eligible_row_idx = np.where(eligible_mask)[0]

    # Sort by |t_stat| descending (most informative first)
    abs_t = np.abs(ref_df["t_stat"].values[eligible_row_idx])
    order = np.argsort(-abs_t)
    return eligible_row_idx[order].astype(np.int32)


# ---------------------------------------------------------------------------
# Convenience: save / load reference arrays (numpy .npy format)
# ---------------------------------------------------------------------------

def save_reference(ref_df: pd.DataFrame, out_dir: Path) -> None:
    """Save a reference DataFrame as compressed numpy arrays.

    Saves one ``.npy`` file per column under ``out_dir``.  This matches
    the GCS upload convention used in the raw notebook.

    Parameters
    ----------
    ref_df : pd.DataFrame
        Reference DataFrame from :func:`construct_reference_from_betas`.
    out_dir : Path
        Directory to write arrays into (created if absent).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for col in ref_df.columns:
        np.save(out_dir / f"{col}.npy", ref_df[col].values)
    print(f"[reference] saved {len(ref_df.columns)} arrays to {out_dir}")


def load_reference(ref_dir: Path) -> pd.DataFrame:
    """Load a reference previously saved by :func:`save_reference`.

    Parameters
    ----------
    ref_dir : Path
        Directory containing the ``.npy`` array files.

    Returns
    -------
    pd.DataFrame
        Reconstructed reference DataFrame.
    """
    ref_dir = Path(ref_dir)
    cols = ["site_index", "N", "slope", "intercept", "R", "t_stat", "P_value"]
    data = {c: np.load(ref_dir / f"{c}.npy") for c in cols}
    return pd.DataFrame(data)
