#!/usr/bin/env python3
"""
submit_dsub_sort.py
~~~~~~~~~~~~~~~~~~~
Submit digital cell-type sorting jobs to Google Cloud Life Sciences (dsub)
for the All of Us long-read methylation dataset.

Each job processes one sample's PAT file through the LongReadClock
single-pass multi-class LLR sorting algorithm and produces seven sorted
PAT + beta files (one per blood cell population).

Usage
-----
    python scripts/submit_dsub_sort.py \\
        --batches broad_revio_batch bcm_revio_batch \\
        --max-samples 5

Requirements
------------
    pip install dsub
    gcloud auth application-default login
"""

import os
import subprocess
import argparse
from pathlib import Path

WORKSPACE_BUCKET = os.environ.get("WORKSPACE_BUCKET", "").rstrip("/")
GOOGLE_PROJECT   = os.environ.get("GOOGLE_PROJECT", "")

# Cell types to sort into
CELL_TYPES = ["Myeloid", "Lymphoid", "T_Cell", "B_Cell", "NK_Cell", "Monocyte", "Granulocyte"]

# Docker image — should have longreadclock, wgbstools, bgzip installed
DOCKER_IMAGE = "gcr.io/google.com/cloudsdktool/cloud-sdk:latest"

TASK_SCRIPT = """\
#!/bin/bash
set -euo pipefail
pip install longreadclock --quiet

# Download input PAT
gsutil -m cp "${IN_PAT_GCS}" "${SAMPLE_KEY}.pat.gz"

# Download marker BEDs
mkdir -p markers
for CT in Myeloid Lymphoid T_Cell B_Cell NK_Cell Monocyte Granulocyte; do
    gsutil cp "${MARKERS_GCS_PREFIX}/Markers.${CT}.bed" "markers/"
done

# Download CpG coordinate file
mkdir -p wgbs_tools/references/hg38
gsutil cp "${CPG_CHROME_SIZE_GCS}" wgbs_tools/references/hg38/CpG.chrome.size

# Download wgbstools binary
gsutil cp "${WGBSTOOLS_BIN_GCS}" ./wgbstools
chmod +x ./wgbstools

# Run sorting
python3 - <<'PYEOF'
import os, subprocess
from pathlib import Path
from longreadclock.preprocessing import load_cpg_chrom_sizes, bgzip_file, wgbstools_index, pat2beta
from longreadclock.sorting import build_chrom_params_target_bg, split_pat_single_pass
from longreadclock.utils import cleanup_files

sample_key = os.environ["SAMPLE_KEY"]
workspace  = os.environ["WORKSPACE_BUCKET"]
tool_bin   = Path("wgbstools")

cs, counts, offsets = load_cpg_chrom_sizes(Path("wgbs_tools/references/hg38/CpG.chrome.size"))
cell_types = ["Myeloid","Lymphoid","T_Cell","B_Cell","NK_Cell","Monocyte","Granulocyte"]

params_by_ct = {}
for ct in cell_types:
    bed = Path(f"markers/Markers.{ct}.bed")
    p, _, _ = build_chrom_params_target_bg(bed, counts, offsets)
    params_by_ct[ct] = p

sorted_paths = split_pat_single_pass(
    Path(f"{sample_key}.pat.gz"),
    Path(sample_key),
    counts, offsets,
    params_by_ct,
    tau=1.053, min_hits=5,
)

for ct, pat_path in sorted_paths.items():
    if not pat_path.exists() or pat_path.stat().st_size == 0:
        continue
    gz  = bgzip_file(pat_path)
    csi = wgbstools_index(gz, tool_bin=str(tool_bin))
    bet = pat2beta(gz, Path(f"betas/{ct}"), genome="hg38",
                   threads=4, tool_bin=str(tool_bin))
    dst = f"{workspace}/results/cell_sorted/{ct}/"
    subprocess.run(["gsutil","-m","-q","cp",str(gz),str(csi),str(bet),dst], check=True)
    cleanup_files([gz, csi, Path(f"betas/{ct}")])

cleanup_files([Path(f"{sample_key}.pat.gz")])
print("Sorting complete:", sample_key)
PYEOF
"""


def list_pat_files(batch_token: str) -> list:
    prefix = f"{WORKSPACE_BUCKET}/results/{batch_token}"
    try:
        raw = subprocess.check_output(
            ["gsutil", "ls", f"{prefix}/*.pat.gz"], text=True
        )
        return [l.strip() for l in raw.splitlines() if l.strip()]
    except subprocess.CalledProcessError:
        return []


def output_exists(sample_key: str, cell_type: str = "Myeloid") -> bool:
    path = f"{WORKSPACE_BUCKET}/results/cell_sorted/{cell_type}/{sample_key}_{cell_type}.pat.gz"
    return subprocess.run(
        ["gsutil", "-q", "stat", path]
    ).returncode == 0


def submit_sort_jobs(
    batch_tokens:  list,
    max_samples:   int = None,
    region:        str = "us-central1",
    skip_existing: bool = True,
):
    tsv_path    = Path("/tmp/sort_tasks.tsv")
    script_path = Path("/tmp/sort_task.sh")
    script_path.write_text(TASK_SCRIPT)
    script_path.chmod(0o755)

    markers_prefix  = f"{WORKSPACE_BUCKET}/results/markers/markers_S2"
    cpg_size_gcs    = f"{WORKSPACE_BUCKET}/resources/CpG.chrome.size"
    wgbstools_bin   = f"{WORKSPACE_BUCKET}/resources/wgbstools"

    rows = []
    for batch in batch_tokens:
        pats = list_pat_files(batch)
        print(f"[{batch}] {len(pats)} PAT files")
        for pat_gcs in pats:
            # sample_key format: {batch}__{person_id}
            person_id  = Path(pat_gcs).stem.replace(".pat", "")
            sample_key = f"{batch}__{person_id}"
            if skip_existing and output_exists(sample_key):
                print(f"  Skipping {sample_key}")
                continue
            rows.append((sample_key, pat_gcs))

    if max_samples:
        rows = rows[:max_samples]

    with open(tsv_path, "w") as f:
        f.write("--env SAMPLE_KEY\t--env IN_PAT_GCS\t--env WORKSPACE_BUCKET"
                "\t--env MARKERS_GCS_PREFIX\t--env CPG_CHROME_SIZE_GCS"
                "\t--env WGBSTOOLS_BIN_GCS\n")
        for sample_key, pat_gcs in rows:
            f.write(f"{sample_key}\t{pat_gcs}\t{WORKSPACE_BUCKET}"
                    f"\t{markers_prefix}\t{cpg_size_gcs}\t{wgbstools_bin}\n")

    print(f"\nSubmitting {len(rows)} sorting jobs via dsub ...")
    cmd = [
        "dsub",
        "--provider",  "google-cls-v2",
        "--project",   GOOGLE_PROJECT,
        "--regions",   region,
        "--logging",   f"{WORKSPACE_BUCKET}/logs/dsub/cell_sorting/",
        "--image",     DOCKER_IMAGE,
        "--script",    str(script_path),
        "--tasks",     str(tsv_path),
        "--name",      "lrc-cell-sorting",
        "--min-ram",   "32",
        "--min-cores", "4",
        "--disk-size", "100",
        "--preemptible",
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Submit LongReadClock cell-type sorting dsub jobs."
    )
    parser.add_argument("--batches", nargs="+", required=True,
                        help="Batch tokens to sort (e.g. broad_revio_batch dsub_JHU_ONT)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--region",      default="us-central1")
    parser.add_argument("--no-skip",     action="store_true")
    args = parser.parse_args()

    assert WORKSPACE_BUCKET, "WORKSPACE_BUCKET not set."
    assert GOOGLE_PROJECT,   "GOOGLE_PROJECT not set."

    submit_sort_jobs(
        batch_tokens  = args.batches,
        max_samples   = args.max_samples,
        region        = args.region,
        skip_existing = not args.no_skip,
    )


if __name__ == "__main__":
    main()
