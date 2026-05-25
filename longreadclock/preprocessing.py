"""
preprocessing.py
~~~~~~~~~~~~~~~~
Core preprocessing utilities for the LongReadClock pipeline.

Covers:
  - Loading the wgbstools CpG chromosome coordinate index
  - PAT file parsing helpers (base inference, coordinate mapping)
  - Marker BED interval → local array slice mapping
  - bgzip compression, wgbstools indexing, PAT → beta conversion

All functions here are extracted directly from the raw analysis notebooks
(cell_sorting.ipynb, preprocess_read_level_methylation.ipynb) and are the
authoritative implementations used by the rest of the package.
"""

import os
import math
import gzip
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reference index loading
# ---------------------------------------------------------------------------

def load_cpg_chrom_sizes(cpg_chrome_size_path: Path) -> Tuple[pd.DataFrame, dict, dict]:
    """Load wgbstools CpG.chrome.size and return coordinate lookup tables.

    Parameters
    ----------
    cpg_chrome_size_path : Path
        Path to the ``CpG.chrome.size`` file produced by wgbstools for hg38.

    Returns
    -------
    cs : pd.DataFrame
        DataFrame with columns ``chrom``, ``n_cpg``, ``offset``.
    chrom_counts : dict
        Mapping of chrom → number of CpGs on that chromosome.
    chrom_offsets : dict
        Mapping of chrom → cumulative CpG offset (0-based start of this
        chromosome in the global CpG array).
    """
    cs = pd.read_csv(cpg_chrome_size_path, sep="\t", header=None,
                     names=["chrom", "n_cpg"])
    cs["n_cpg"] = cs["n_cpg"].astype(np.int64)
    cs["offset"] = cs["n_cpg"].cumsum().shift(fill_value=0).astype(np.int64)

    chrom_counts = dict(zip(cs["chrom"], cs["n_cpg"]))
    chrom_offsets = dict(zip(cs["chrom"], cs["offset"]))
    return cs, chrom_counts, chrom_offsets


# ---------------------------------------------------------------------------
# PAT coordinate helpers
# ---------------------------------------------------------------------------

def infer_pat_base(pat_gz: Path, n: int = 20_000) -> int:
    """Infer whether a PAT file uses 0-based or 1-based CpG coordinates.

    Reads up to *n* lines and returns 1 if the minimum observed start
    position is ≥ 1 (1-based), otherwise 0 (0-based).

    Parameters
    ----------
    pat_gz : Path
        Path to a gzip-compressed PAT file.
    n : int
        Maximum number of lines to inspect.
    """
    mn = math.inf
    with gzip.open(pat_gz, "rt") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                mn = min(mn, int(parts[1]))
            except ValueError:
                pass
    return 1 if mn >= 1 else 0


def pat_start_to_local_j(
    chrom: str,
    start_cpg_1based: int,
    chrom_counts: dict,
    chrom_offsets: dict,
    base: int = 1,
) -> Optional[int]:
    """Convert a PAT file CpG start position to a 0-based local array index.

    The PAT format can encode positions either as chromosome-local 1-based
    indices or as global (genome-wide) 1-based indices.  This function
    handles both cases transparently.

    Parameters
    ----------
    chrom : str
        Chromosome name (e.g. ``"chr1"``).
    start_cpg_1based : int
        The CpG start coordinate as it appears in the PAT file.
    chrom_counts : dict
        From :func:`load_cpg_chrom_sizes`.
    chrom_offsets : dict
        From :func:`load_cpg_chrom_sizes`.
    base : int
        Coordinate base reported by :func:`infer_pat_base` (0 or 1).

    Returns
    -------
    int or None
        0-based local index into the chromosome CpG array, or ``None`` if
        the coordinate is out of range.
    """
    if chrom not in chrom_counts:
        return None

    n = int(chrom_counts[chrom])
    off = int(chrom_offsets[chrom])

    # Accept either chromosome-local or global coordinates
    if 1 <= start_cpg_1based <= n:
        local_1based = start_cpg_1based
    elif off + 1 <= start_cpg_1based <= off + n:
        local_1based = start_cpg_1based - off
    else:
        return None

    j = int(local_1based - base)
    if j < 0 or j >= n:
        return None
    return j


def marker_interval_to_local_slice(
    chrom: str,
    cpg_start: int,
    cpg_end: int,
    chrom_counts: dict,
    chrom_offsets: dict,
) -> Optional[Tuple[int, int]]:
    """Map a marker BED CpG interval to a (j0, j1) local array slice.

    The marker BED files produced by wgbstools use 1-based CpG coordinates
    and may encode them as either chromosome-local or global positions.
    This function resolves both cases and returns a half-open slice
    ``[j0, j1)`` into the per-chromosome CpG probability array.

    Parameters
    ----------
    chrom : str
        Chromosome name.
    cpg_start : int
        1-based start CpG coordinate from the marker BED.
    cpg_end : int
        1-based end CpG coordinate (exclusive) from the marker BED.
    chrom_counts, chrom_offsets : dict
        From :func:`load_cpg_chrom_sizes`.

    Returns
    -------
    (j0, j1) or None
        Half-open slice, or ``None`` if the interval is invalid.
    """
    if chrom not in chrom_counts:
        return None

    n = int(chrom_counts[chrom])
    off = int(chrom_offsets[chrom])

    # Determine if coordinates are local or global
    if 1 <= cpg_start <= n and 1 <= cpg_end <= n + 1:
        local_start = cpg_start
        local_end = cpg_end
    elif off + 1 <= cpg_start <= off + n and off + 1 <= cpg_end <= off + n + 1:
        local_start = cpg_start - off
        local_end = cpg_end - off
    else:
        return None

    j0 = max(0, min(int(local_start) - 1, n))
    j1 = max(0, min(int(local_end) - 1, n))
    if j1 <= j0:
        return None
    return j0, j1


# ---------------------------------------------------------------------------
# wgbstools tool wrappers
# ---------------------------------------------------------------------------

def _find_wgbstools() -> str:
    """Locate the wgbstools binary; raise if not found."""
    # Prefer a local copy inside the workspace (typical AoU setup)
    candidates = [
        Path(os.getcwd()) / "wgbs_tools" / "wgbstools",
        Path(os.path.expanduser("~")) / "wgbs_tools" / "wgbstools",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Fall back to PATH
    import shutil
    w = shutil.which("wgbstools")
    if w:
        return w
    raise FileNotFoundError(
        "wgbstools binary not found. "
        "Clone https://github.com/nloyfer/wgbs_tools and run `make` inside it, "
        "or set WGBSTOOLS_PATH in your environment."
    )


def bgzip_file(in_txt: Path) -> Path:
    """Compress a plain-text PAT file with bgzip and return the .gz path.

    Parameters
    ----------
    in_txt : Path
        Path to the uncompressed PAT text file.  The file is removed after
        compression.

    Returns
    -------
    Path
        Path to the newly created ``.gz`` file.
    """
    out_gz = Path(str(in_txt) + ".gz")
    if out_gz.exists():
        out_gz.unlink()
    subprocess.run(["bgzip", "-f", str(in_txt)], check=True)
    if not out_gz.exists():
        raise FileNotFoundError(f"bgzip produced no output: expected {out_gz}")
    return out_gz


def wgbstools_index(pat_gz: Path, tool_bin: Optional[str] = None) -> Path:
    """Create a ``.csi`` index for a bgzip-compressed PAT file.

    Parameters
    ----------
    pat_gz : Path
        Path to the ``.pat.gz`` file.
    tool_bin : str, optional
        Explicit path to the wgbstools binary.  Auto-detected if omitted.

    Returns
    -------
    Path
        Path to the ``.pat.gz.csi`` index file.
    """
    if tool_bin is None:
        tool_bin = _find_wgbstools()
    csi = Path(str(pat_gz) + ".csi")
    if csi.exists():
        csi.unlink()
    subprocess.run([tool_bin, "index", str(pat_gz)], check=True)
    if not csi.exists():
        raise FileNotFoundError(f"wgbstools index produced no output: expected {csi}")
    return csi


def pat2beta(
    pat_gz: Path,
    out_dir: Path,
    genome: str = "hg38",
    threads: int = 4,
    force: bool = True,
    tool_bin: Optional[str] = None,
) -> Path:
    """Convert a bgzip-compressed PAT file to a wgbstools beta file.

    Parameters
    ----------
    pat_gz : Path
        Input ``.pat.gz`` file.
    out_dir : Path
        Directory where the beta file will be written.
    genome : str
        Genome build identifier registered with wgbstools (default ``"hg38"``).
    threads : int
        Number of parallel threads.
    force : bool
        Pass ``-f`` to wgbstools to overwrite existing output.
    tool_bin : str, optional
        Explicit path to the wgbstools binary.  Auto-detected if omitted.

    Returns
    -------
    Path
        Path to the produced ``.beta`` file.
    """
    if tool_bin is None:
        tool_bin = _find_wgbstools()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [tool_bin, "pat2beta"]
    if force:
        cmd.append("-f")
    cmd += ["--genome", genome, "--threads", str(threads),
            "--out_dir", str(out_dir), str(pat_gz)]

    subprocess.run(cmd, check=True)

    # Resolve the output path — wgbstools names it after the PAT stem
    stem = pat_gz.name
    if stem.endswith(".pat.gz"):
        stem = stem[:-7]
    cand = out_dir / f"{stem}.beta"
    if cand.exists():
        return cand

    # Fall back to the most recently modified beta file in out_dir
    betas = sorted(out_dir.glob("*.beta"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if betas:
        return betas[0]

    raise FileNotFoundError(
        f"pat2beta produced no .beta file in {out_dir}"
    )
