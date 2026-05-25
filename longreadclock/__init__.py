"""
LongReadClock
~~~~~~~~~~~~~
Cell-type-resolved epigenetic aging clocks from bulk long-read methylation data.

Pipeline:
  preprocessing → QC → cell-type sorting → age reference → age inference

Usage on All of Us Workbench:
  pip install -e /home/jupyter/LongReadClock_Claude
  from longreadclock.preprocessing import load_cpg_chrom_sizes
  from longreadclock.sorting import build_chrom_params_target_bg, split_pat_single_pass
  from longreadclock.reference import construct_reference_from_betas, select_eligible_sites
  from longreadclock.inference import predict_age_bernoulli_grid, run_5fold_cross_validation
"""

__version__ = "1.0.0"
__author__  = "Yinjie Wu, Mahdi Moqri, Vadim N. Gladyshev"

# Expose the most commonly imported symbols at package level
from .preprocessing import (
    load_cpg_chrom_sizes,
    infer_pat_base,
    pat_start_to_local_j,
    marker_interval_to_local_slice,
    bgzip_file,
    wgbstools_index,
    pat2beta,
)
from .utils import (
    get_workspace_bucket,
    safe_label,
    gcs_exists,
    gsutil_ls,
    gcs_copy,
    make_sample_key,
    cleanup_files,
    quantile_from_hist,
)
from .sorting import (
    build_chrom_params_target_bg,
    llr_for_row_sparse,
    split_pat_single_pass,
)
from .reference import (
    construct_reference_from_betas,
    select_eligible_sites,
    save_reference,
    load_reference,
)
from .inference import (
    predict_age_bernoulli_grid,
    run_5fold_cross_validation,
    run_cross_platform_benchmarks,
)
from .qc import (
    calculate_ambiguous_fraction,
    calculate_mean_binary_entropy,
    build_pca_matrix,
    run_pca,
    run_elovl2_validation,
    ELOVL2_CPG_INDEX,
)
