"""
utils.py
~~~~~~~~
General GCS / filesystem utilities shared across the LongReadClock pipeline.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from collections import Counter
from typing import List, Optional


def get_workspace_bucket() -> str:
    """Return the All of Us workspace GCS bucket from the environment.

    Falls back to a clearly labelled placeholder if not running inside
    an All of Us Workbench notebook.
    """
    bucket = os.environ.get("WORKSPACE_BUCKET", "")
    if not bucket:
        bucket = "gs://fc-secure-PLACEHOLDER-bucket"
    return bucket.rstrip("/")


def resolve_path(path_str: str) -> str:
    """Substitute ``{WORKSPACE_BUCKET}`` placeholders in a path string."""
    return path_str.replace("{WORKSPACE_BUCKET}", get_workspace_bucket())


def safe_label(s: str) -> str:
    """Sanitize a string for use in filenames (replace non-alphanumeric with ``_``)."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)


def run_cmd(
    args: List[str],
    check: bool = True,
    text:  bool = True,
    capture_output: bool = False,
):
    """Run a subprocess command, optionally capturing output."""
    try:
        if capture_output:
            result = subprocess.run(args, check=check, capture_output=True, text=text)
            return result.stdout.strip()
        else:
            subprocess.run(args, check=check)
    except subprocess.CalledProcessError as exc:
        print(f"Error running: {' '.join(args)}\n{exc}")
        raise


def gcs_exists(path: str) -> bool:
    """Return True if a GCS object exists (uses gsutil stat)."""
    path = resolve_path(path)
    try:
        subprocess.run(["gsutil", "-q", "stat", path], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def gsutil_ls(prefix: str) -> List[str]:
    """List GCS objects under a prefix; returns an empty list on error."""
    prefix = resolve_path(prefix)
    try:
        out = subprocess.check_output(["gsutil", "ls", prefix], text=True)
        return [x.strip() for x in out.splitlines() if x.strip()]
    except subprocess.CalledProcessError:
        return []


def gcs_copy(src: str, dst: str) -> None:
    """Copy a file to/from GCS using ``gsutil -m cp``."""
    src = resolve_path(src)
    dst = resolve_path(dst)
    print(f"Copying {src} → {dst}")
    run_cmd(["gsutil", "-m", "cp", src, dst])


def make_sample_key(pat_gcs_path: str) -> str:
    """Construct a canonical sample key from a GCS PAT path.

    Format: ``{parent_folder}__{sample_id}``

    Example: ``gs://bucket/results/broad_revio_batch/1001075.pat.gz``
    → ``broad_revio_batch__1001075``
    """
    p         = Path(pat_gcs_path)
    sample_id = p.name.replace(".pat.gz", "")
    parent    = p.parent.name
    return f"{parent}__{sample_id}"


def cleanup_files(paths) -> None:
    """Delete a list of local files or directories, silently skipping errors."""
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        except Exception as exc:
            print(f"Warning: cleanup failed for {p}: {exc}")


def quantile_from_hist(hist: Counter, q: float) -> float:
    """Compute quantile ``q`` from a Counter histogram.

    Used to compute median / P90 hit counts from read scoring statistics
    without materialising the full hit-count array.
    """
    if not hist:
        return 0.0
    items = sorted(hist.items())
    total = sum(c for _, c in items)
    if total == 0:
        return 0.0
    thresh = q * (total - 1)
    cum    = 0
    for val, cnt in items:
        if cum + cnt > thresh:
            return float(val)
        cum += cnt
    return float(items[-1][0])
