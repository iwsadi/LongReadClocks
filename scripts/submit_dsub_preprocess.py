#!/usr/bin/env python3
"""
submit_dsub_preprocess.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Submit wgbstools BAM→PAT preprocessing jobs to Google Cloud Life Sciences
(GCLS / dsub) for the All of Us long-read whole-genome sequencing dataset.

dsub runs one Docker container per sample in parallel on Google Cloud,
downloading the BAM, running wgbstools bam2pat, and uploading the PAT/beta
outputs to GCS.

Usage
-----
    python scripts/submit_dsub_preprocess.py \\
        --batch broad_revio_batch \\
        --bam-dir gs://fc-aou-datasets-controlled/pooled/longreads/v8_delta/Broad/revio/bam \\
        --max-samples 10

Requirements
------------
    pip install dsub
    gcloud auth application-default login
"""

import os
import subprocess
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE_BUCKET = os.environ.get("WORKSPACE_BUCKET", "").rstrip("/")
GOOGLE_PROJECT   = os.environ.get("GOOGLE_PROJECT", "")

# Docker image with wgbstools + samtools pre-installed
# Build and push your own, or use the shared AoU image if available.
DOCKER_IMAGE = "gcr.io/google.com/cloudsdktool/cloud-sdk:latest"

# dsub task script written to the worker
TASK_SCRIPT = """\
#!/bin/bash
set -euo pipefail

# Download BAM
echo "Downloading ${PERSON_ID}.bam..."
gcloud storage cp "${BAM_GCS_PATH}" "${PERSON_ID}.bam" \\
    --billing-project="${GOOGLE_PROJECT}"

# Convert BAM → PAT (+ beta as side effect)
echo "Running bam2pat..."
/home/wgbstools/wgbstools bam2pat "${PERSON_ID}.bam" --genome hg38 -f

# Upload outputs
echo "Uploading to GCS..."
gcloud storage cp "${PERSON_ID}.pat.gz" "${OUT_PREFIX}/" 2>/dev/null || true
gcloud storage cp "${PERSON_ID}.beta"   "${OUT_PREFIX}/" 2>/dev/null || true

# Clean up
rm -f "${PERSON_ID}.bam" "${PERSON_ID}.pat.gz" "${PERSON_ID}.beta"
echo "Done: ${PERSON_ID}"
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def list_bam_files(bam_dir: str, project: str) -> list:
    """List all GRCh38 BAM files in a GCS directory."""
    try:
        raw = subprocess.check_output(
            f"gsutil -u {project} ls {bam_dir}/**.bam",
            shell=True, text=True
        )
        return [
            line.strip() for line in raw.splitlines()
            if line.strip().endswith(".bam") and "/GRCh38/" in line
        ]
    except subprocess.CalledProcessError:
        return []


def output_exists(person_id: str, out_prefix: str) -> bool:
    """Return True if the PAT output already exists in GCS."""
    result = subprocess.run(
        f"gsutil -q stat {out_prefix}/{person_id}.pat.gz",
        shell=True
    )
    return result.returncode == 0


def submit_dsub_job(
    batch_name:  str,
    bam_files:   list,
    out_prefix:  str,
    project:     str,
    region:      str = "us-central1",
    skip_existing: bool = True,
):
    """Build and submit a dsub job array for the given BAM files."""
    # Write the task TSV
    tsv_path = Path(f"/tmp/{batch_name}_tasks.tsv")
    with open(tsv_path, "w") as f:
        f.write("--env PERSON_ID\t--env BAM_GCS_PATH\t--env OUT_PREFIX\t--env GOOGLE_PROJECT\n")
        for bam_path in bam_files:
            person_id = Path(bam_path).stem
            if skip_existing and output_exists(person_id, out_prefix):
                print(f"  Skipping {person_id} (already done)")
                continue
            f.write(f"{person_id}\t{bam_path}\t{out_prefix}\t{project}\n")

    # Write task script
    script_path = Path(f"/tmp/{batch_name}_task.sh")
    script_path.write_text(TASK_SCRIPT)
    script_path.chmod(0o755)

    # Submit via dsub
    cmd = [
        "dsub",
        "--provider",    "google-cls-v2",
        "--project",     project,
        "--regions",     region,
        "--logging",     f"{WORKSPACE_BUCKET}/logs/dsub/{batch_name}/",
        "--image",       DOCKER_IMAGE,
        "--script",      str(script_path),
        "--tasks",       str(tsv_path),
        "--name",        f"lrc-preprocess-{batch_name}",
        "--min-ram",     "16",
        "--min-cores",   "4",
        "--disk-size",   "200",
        "--preemptible",
    ]
    print(f"\nSubmitting dsub job: {batch_name}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Submit wgbstools preprocessing dsub jobs for All of Us LR-WGS data."
    )
    parser.add_argument("--batch",       required=True,
                        help="Output folder name (e.g. broad_revio_batch)")
    parser.add_argument("--bam-dir",     required=True,
                        help="GCS directory containing BAM files")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to N samples (for testing)")
    parser.add_argument("--region",      default="us-central1",
                        help="GCP region for dsub jobs")
    parser.add_argument("--no-skip",     action="store_true",
                        help="Re-process samples even if output exists")
    args = parser.parse_args()

    assert WORKSPACE_BUCKET, "WORKSPACE_BUCKET environment variable not set."
    assert GOOGLE_PROJECT,   "GOOGLE_PROJECT environment variable not set."

    out_prefix = f"{WORKSPACE_BUCKET}/results/{args.batch}"

    print(f"Scanning BAMs in: {args.bam_dir}")
    bam_files = list_bam_files(args.bam_dir, GOOGLE_PROJECT)
    print(f"Found {len(bam_files)} BAMs")

    if args.max_samples:
        bam_files = bam_files[:args.max_samples]
        print(f"Limited to {len(bam_files)} samples")

    submit_dsub_job(
        batch_name    = args.batch,
        bam_files     = bam_files,
        out_prefix    = out_prefix,
        project       = GOOGLE_PROJECT,
        region        = args.region,
        skip_existing = not args.no_skip,
    )


if __name__ == "__main__":
    main()
