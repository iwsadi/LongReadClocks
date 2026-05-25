"""
sorting.py
~~~~~~~~~~
Read-level digital cell-type sorting for LongReadClock.

Each sequencing read in a PAT file is scored against purified-cell WGBS
marker reference profiles using a log-likelihood ratio (LLR) approach.
Reads whose LLR exceeds a threshold ``tau`` are assigned to that cell type.

This module implements the **single-pass multi-class** strategy used in the
published analysis: all seven target cell types are evaluated simultaneously
in one pass through each PAT file, so each read can be assigned to the
cell type it best matches.

The seven targets are:
    Coarse lineages: Myeloid, Lymphoid
    Fine subtypes:   T_Cell, B_Cell, NK_Cell, Monocyte, Granulocyte

Extracted faithfully from ``cell_sorting.ipynb`` (raw notebook).
"""

import re
import math
import gzip
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .preprocessing import (
    marker_interval_to_local_slice,
    pat_start_to_local_j,
    infer_pat_base,
)

# ---------------------------------------------------------------------------
# PAT line parser
# ---------------------------------------------------------------------------

PAT_LINE_RE = re.compile(r"^(\S+)\t(\d+)\t(\S+)\t(\d+)\s*$")
METH_CHARS   = frozenset(("C", "1"))
UNMETH_CHARS = frozenset(("T", "0"))


def parse_pat_line(line: str) -> Optional[Tuple[str, int, str, int]]:
    """Parse a single PAT file line.

    Returns
    -------
    (chrom, start_cpg, pattern, multiplicity) or None if malformed.
    """
    m = PAT_LINE_RE.match(line)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))


# ---------------------------------------------------------------------------
# Marker parameter loading
# ---------------------------------------------------------------------------

def build_chrom_params_target_bg(
    markers_bed: Path,
    chrom_counts: dict,
    chrom_offsets: dict,
    eps: float = 1e-4,
) -> Tuple[dict, int, tuple]:
    """Load a marker BED file and build per-chromosome target/background arrays.

    Each row in the BED encodes a CpG interval with a target cell-type
    methylation probability (column 10) and a background probability
    (column 11).  These are stored as float32 arrays indexed by local CpG
    position within each chromosome for fast lookup during read scoring.

    Parameters
    ----------
    markers_bed : Path
        Path to a marker BED file (wgbstools format, ≥ 11 columns).
    chrom_counts, chrom_offsets : dict
        From :func:`~longreadclock.preprocessing.load_cpg_chrom_sizes`.
    eps : float
        Clipping epsilon applied to probabilities (avoids log(0)).

    Returns
    -------
    chrom_params : dict
        Mapping chrom → {``"p_tg"``: np.ndarray, ``"p_bg"``: np.ndarray}.
    total_marker_sites : int
        Total number of finite marker sites loaded.
    bed_shape : tuple
        Shape of the raw BED DataFrame (for diagnostics).
    """
    df = pd.read_csv(markers_bed, sep="\t", comment="#", header=None)
    if df.shape[1] < 11:
        raise ValueError(
            f"Marker BED has too few columns: n_cols={df.shape[1]}. "
            "Expected ≥ 11 (wgbstools format with p_tg at col 10, p_bg at col 11)."
        )

    use = df[[0, 3, 4, 9, 10]].copy()
    use.columns = ["chrom", "cpg_start", "cpg_end", "p_tg", "p_bg"]
    use["cpg_start"] = use["cpg_start"].astype(np.int64)
    use["cpg_end"]   = use["cpg_end"].astype(np.int64)
    use["p_tg"]      = use["p_tg"].astype(np.float32)
    use["p_bg"]      = use["p_bg"].astype(np.float32)

    # Initialise NaN arrays for every chromosome
    chrom_params: Dict[str, dict] = {}
    for chrom, n in chrom_counts.items():
        chrom_params[chrom] = {
            "p_tg": np.full(int(n), np.nan, dtype=np.float32),
            "p_bg": np.full(int(n), np.nan, dtype=np.float32),
        }

    for chrom, g in use.groupby("chrom"):
        if chrom not in chrom_params:
            continue
        p_tg_arr = chrom_params[chrom]["p_tg"]
        p_bg_arr = chrom_params[chrom]["p_bg"]

        for cpg_s, cpg_e, ptg, pbg in zip(
            g["cpg_start"].values,
            g["cpg_end"].values,
            g["p_tg"].values,
            g["p_bg"].values,
        ):
            sl = marker_interval_to_local_slice(
                chrom, int(cpg_s), int(cpg_e), chrom_counts, chrom_offsets
            )
            if sl is None:
                continue
            j0, j1 = sl
            p_tg_arr[j0:j1] = float(np.clip(ptg, eps, 1.0 - eps))
            p_bg_arr[j0:j1] = float(np.clip(pbg, eps, 1.0 - eps))

    total_marker_sites = sum(
        int(np.isfinite(chrom_params[c]["p_tg"]).sum()) for c in chrom_params
    )
    return chrom_params, total_marker_sites, df.shape


# ---------------------------------------------------------------------------
# LLR scoring
# ---------------------------------------------------------------------------

def llr_for_row_sparse(
    pattern: str,
    p_tg_seg: np.ndarray,
    p_bg_seg: np.ndarray,
    eps: float = 1e-4,
) -> Tuple[float, int]:
    """Compute the log-likelihood ratio for a single read pattern.

    Only CpG positions that have finite marker probabilities (i.e. are
    covered by the marker BED) contribute to the score.

    Parameters
    ----------
    pattern : str
        The CpG pattern string from the PAT line
        (``C``/``1`` = methylated, ``T``/``0`` = unmethylated, other = skip).
    p_tg_seg : np.ndarray
        Target cell-type methylation probability array for this read's span.
    p_bg_seg : np.ndarray
        Background methylation probability array for the same span.
    eps : float
        Probability clipping to avoid numerical underflow.

    Returns
    -------
    (llr, hits) : (float, int)
        ``llr`` — cumulative log-likelihood ratio for this read.
        ``hits`` — number of informative CpG positions used.
    """
    mask = np.isfinite(p_tg_seg) & np.isfinite(p_bg_seg)
    if not mask.any():
        return 0.0, 0

    idxs = np.flatnonzero(mask)
    pt = np.clip(p_tg_seg[idxs], eps, 1.0 - eps)
    pb = np.clip(p_bg_seg[idxs], eps, 1.0 - eps)

    log_m = np.log(pt) - np.log(pb)
    log_u = np.log1p(-pt) - np.log1p(-pb)

    llr  = 0.0
    hits = 0
    for k, i in enumerate(idxs):
        ch = pattern[i]
        if ch in METH_CHARS:
            llr  += float(log_m[k])
            hits += 1
        elif ch in UNMETH_CHARS:
            llr  += float(log_u[k])
            hits += 1

    return llr, hits


# ---------------------------------------------------------------------------
# Single-pass multi-class read sorting
# ---------------------------------------------------------------------------

def split_pat_single_pass(
    in_pat_gz: Path,
    out_prefix: Path,
    chrom_counts: dict,
    chrom_offsets: dict,
    all_target_params: dict,
    tau: float = 1.053,
    min_hits: int = 5,
) -> Dict[str, Path]:
    """Assign reads from a PAT file to cell types in a single pass.

    Each read is scored against all target cell types simultaneously.
    A read is assigned to a target if:
      1. Its LLR for that target exceeds ``log(tau)``, and
      2. It has the highest LLR among all qualifying targets
         (with a minimum gap of 0.1 to resolve near-ties).

    Reads that meet no threshold or produce an ambiguous tie are discarded.

    This is the production algorithm used in the published LongReadClock
    analysis.  It exactly reproduces ``process_one_sample_all7()`` from
    ``cell_sorting.ipynb``.

    Parameters
    ----------
    in_pat_gz : Path
        Path to the gzip-compressed input PAT file.
    out_prefix : Path
        Output path prefix.  Each cell type writes to
        ``{out_prefix}_{target}.pat``.
    chrom_counts, chrom_offsets : dict
        From :func:`~longreadclock.preprocessing.load_cpg_chrom_sizes`.
    all_target_params : dict
        Mapping target name → chrom_params dict
        (from :func:`build_chrom_params_target_bg`).
    tau : float
        LLR threshold in probability-ratio space (default 1.053, i.e.
        ``log(1.053) ≈ 0.052``).  Equivalent to ~5 % more likely under
        the target model.
    min_hits : int
        Minimum number of informative CpGs a read must cover to be scored.

    Returns
    -------
    dict
        Mapping target name → Path of the output ``.pat`` file.
    """
    base    = infer_pat_base(in_pat_gz)
    tau_log = math.log(tau)

    out_fh: Dict[str, object] = {}
    out_paths: Dict[str, Path] = {}

    for target in all_target_params:
        p = Path(str(out_prefix) + f"_{target}.pat")
        if p.exists():
            p.unlink()
        out_paths[target] = p
        out_fh[target]    = open(p, "wt")

    counts: Dict[str, dict] = {
        t: {"Assigned": 0, "LowHits": 0, "LowLLR": 0, "NoOverlap": 0}
        for t in all_target_params
    }
    counts["Skipped"]      = 0
    counts["AmbiguousTie"] = 0

    with gzip.open(in_pat_gz, "rt") as fin:
        for line in fin:
            parsed = parse_pat_line(line)
            if not parsed:
                counts["Skipped"] += 1
                continue

            chrom, start_cpg, pattern, n = parsed

            if chrom not in chrom_counts:
                counts["Skipped"] += n
                continue

            j = pat_start_to_local_j(
                chrom, start_cpg, chrom_counts, chrom_offsets, base=base
            )
            if j is None:
                counts["Skipped"] += n
                continue

            L = len(pattern)
            if j + L > int(chrom_counts[chrom]):
                counts["Skipped"] += n
                continue

            # Score against every target
            scores: List[Tuple[str, float, int]] = []
            for target, params in all_target_params.items():
                p_tg = params[chrom]["p_tg"][j : j + L]
                p_bg = params[chrom]["p_bg"][j : j + L]
                llr, hits = llr_for_row_sparse(pattern, p_tg, p_bg)
                scores.append((target, llr, hits))

            # Keep targets that meet threshold and hit count
            valid: List[Tuple[str, float]] = [
                (t, llr)
                for t, llr, hits in scores
                if hits >= min_hits and llr > tau_log
            ]

            if not valid:
                # Track per-target reasons
                for t, llr, hits in scores:
                    if hits == 0:
                        counts[t]["NoOverlap"] += n
                    elif hits < min_hits:
                        counts[t]["LowHits"] += n
                    else:
                        counts[t]["LowLLR"] += n
                continue

            if len(valid) == 1:
                assigned = valid[0][0]
                out_fh[assigned].write(line)
                counts[assigned]["Assigned"] += n
            else:
                # Assign to highest LLR; discard if near-tie (< 0.1 gap)
                valid.sort(key=lambda x: x[1], reverse=True)
                if valid[0][1] - valid[1][1] < 0.1:
                    counts["AmbiguousTie"] += n
                else:
                    assigned = valid[0][0]
                    out_fh[assigned].write(line)
                    counts[assigned]["Assigned"] += n

    for fh in out_fh.values():
        fh.close()

    print(
        f"[sorting] single-pass complete | "
        f"assigned={sum(counts[t]['Assigned'] for t in all_target_params)} | "
        f"ambiguous={counts['AmbiguousTie']} | "
        f"skipped={counts['Skipped']}"
    )
    return out_paths
