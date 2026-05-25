"""
validation.py
~~~~~~~~~~~~~
DMR-level validation of digital cell-type sorting quality.

After reads have been digitally sorted and converted to beta files, this
module verifies that sorted reads have the expected methylation pattern at
the marker regions used for sorting.  For each cell type, the sorted reads'
methylation profile over that cell type's markers should closely match the
purified reference atlas, while profiles from other cell types should differ.

Mean Absolute Difference (MAD) is used as the comparison metric.  A good
sort produces low MAD for matched cell-type pairs and high MAD for mismatched
pairs, analogous to a confusion matrix in methylation space.

Extracted from ``DMR_validation.ipynb`` (raw notebook).
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .preprocessing import marker_interval_to_local_slice


# ---------------------------------------------------------------------------
# Region-level methylation extraction
# ---------------------------------------------------------------------------

def load_profile_over_regions(
    beta_path: Path,
    region_df: pd.DataFrame,
    chrom_counts: dict,
    chrom_offsets: dict,
    impute_val: float = 0.5,
) -> np.ndarray:
    """Extract average methylation per region from a beta file.

    Parameters
    ----------
    beta_path : Path
        Path to a local ``.beta`` file.
    region_df : pd.DataFrame
        DataFrame with columns ``chrom``, ``cpg_start``, ``cpg_end``
        (1-based CpG coordinates, wgbstools convention).
    chrom_counts, chrom_offsets : dict
        From :func:`~longreadclock.preprocessing.load_cpg_chrom_sizes`.
    impute_val : float
        Value imputed for uncovered CpG sites (default 0.5 = unbiased).

    Returns
    -------
    np.ndarray of float, shape (n_regions,)
        Mean methylation fraction over each region.
    """
    from .reference import _load_beta_pairs_local
    pairs = _load_beta_pairs_local(beta_path)
    meth  = pairs[:, 0].astype(np.float64)
    total = pairs[:, 1].astype(np.float64)

    beta = np.full(len(meth), impute_val, dtype=np.float64)
    cov_mask = total > 0
    beta[cov_mask] = meth[cov_mask] / total[cov_mask]

    region_avgs = []
    for _, row in region_df.iterrows():
        sl = marker_interval_to_local_slice(
            str(row["chrom"]),
            int(row["cpg_start"]),
            int(row["cpg_end"]),
            chrom_counts,
            chrom_offsets,
        )
        if sl is None:
            region_avgs.append(impute_val)
            continue
        j0, j1 = sl
        region_avgs.append(float(np.mean(beta[j0:j1])))

    return np.array(region_avgs, dtype=np.float64)


def avg_profiles_over_regions(
    beta_paths: List[Path],
    region_df: pd.DataFrame,
    chrom_counts: dict,
    chrom_offsets: dict,
) -> np.ndarray:
    """Average region methylation profiles across multiple beta files."""
    profiles = []
    for path in beta_paths:
        try:
            p = load_profile_over_regions(
                path, region_df, chrom_counts, chrom_offsets
            )
            profiles.append(p)
        except Exception as exc:
            print(f"[validation] warning: failed to load {path}: {exc}")
    if not profiles:
        return np.full(len(region_df), 0.5)
    return np.mean(profiles, axis=0)


# ---------------------------------------------------------------------------
# MAD-based validation
# ---------------------------------------------------------------------------

def compute_mad(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """Compute Mean Absolute Difference between two methylation profiles."""
    return float(np.mean(np.abs(profile_a - profile_b)))


def run_dmr_validation_all_celltypes(
    sorted_betas_dict:    Dict[str, List[Path]],
    reference_betas_dict: Dict[str, List[Path]],
    marker_beds_dict:     Dict[str, Path],
    chrom_counts:         dict,
    chrom_offsets:        dict,
) -> pd.DataFrame:
    """Cross-validate cell-type sorting quality using DMR methylation profiles.

    For each assigned cell type, this function computes the mean methylation
    profile over that cell type's marker regions for both:
    - The sorted reads (digitally assigned to that cell type), and
    - Each reference atlas cell type.

    The MAD between each pair (assigned cell type, reference cell type) is
    recorded.  Matched pairs (same cell type) should have low MAD; mismatched
    pairs should have high MAD.

    Parameters
    ----------
    sorted_betas_dict : dict
        Mapping cell-type name → list of sorted ``.beta`` file paths.
    reference_betas_dict : dict
        Mapping cell-type name → list of purified-reference ``.beta`` paths.
    marker_beds_dict : dict
        Mapping cell-type name → path to that cell type's marker BED file.
    chrom_counts, chrom_offsets : dict
        From :func:`~longreadclock.preprocessing.load_cpg_chrom_sizes`.

    Returns
    -------
    pd.DataFrame
        Columns: ``assigned_celltype``, ``reference_celltype``,
        ``mad``, ``is_matched``.
    """
    # Build region DataFrames from marker BEDs
    marker_regions: Dict[str, pd.DataFrame] = {}
    for ct, bed_path in marker_beds_dict.items():
        df = pd.read_csv(bed_path, sep="\t", comment="#", header=None)
        use = df[[0, 3, 4]].copy()
        use.columns = ["chrom", "cpg_start", "cpg_end"]
        marker_regions[ct] = use

    # Average profiles for sorted reads
    sorted_profiles: Dict[str, np.ndarray] = {}
    for ct, paths in sorted_betas_dict.items():
        if ct not in marker_regions:
            continue
        print(f"[validation] computing sorted profile for {ct} ...")
        sorted_profiles[ct] = avg_profiles_over_regions(
            paths, marker_regions[ct], chrom_counts, chrom_offsets
        )

    results = []
    for ct_assigned, p_assigned in sorted_profiles.items():
        reg_df = marker_regions[ct_assigned]
        for ct_ref, ref_paths in reference_betas_dict.items():
            p_ref = avg_profiles_over_regions(
                ref_paths, reg_df, chrom_counts, chrom_offsets
            )
            mad = compute_mad(p_assigned, p_ref)
            results.append({
                "assigned_celltype":   ct_assigned,
                "reference_celltype":  ct_ref,
                "mad":                 mad,
                "is_matched":          ct_assigned == ct_ref,
            })

    return pd.DataFrame(results)
