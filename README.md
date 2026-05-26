# LongReadClocks

**Cell-type-resolved epigenetic aging clocks from bulk long-read methylation data**

> Wu Y, Moqri M, Gladyshev VN. *Measuring Cell Type Resolved Aging Through Long-Read Sequencing.* 2026.

LongReadClocks constructs epigenetic aging clocks for seven blood cell populations — myeloid, lymphoid, T cells, B cells, NK cells, monocytes, and granulocytes — directly from **bulk** long-read whole-genome sequencing data, without physical cell sorting. Individual sequencing reads are digitally assigned to cell lineages using methylation-based likelihood-ratio scoring against purified-cell WGBS reference profiles.


## Pipeline overview

```
BAM files (All of Us LR-WGS)
        |
        v
01  BAM -> PAT -> beta         (wgbstools bam2pat, one sample at a time)
        |
        v
02  Quality control            (ML-score ambiguous fraction, bulk PCA,
        |                       ELOVL2 sanity check)
        v
03  Cell-type reference        (download purified-cell WGBS marker BEDs,
        |                       build target/background probability arrays)
        v
04  Digital sorting            (single-pass LLR read classification ->
        |                       7 sorted PAT + beta files per sample)
        v
05  Age reference construction (incremental OLS regression of methylation
        |                       vs age, per CpG, per cell-type compartment)
        v
06  Age prediction             (Bernoulli grid search, top-1000 CpGs,
        |                       8x8 cross-reference evaluation)
        v
07  Phenotype analysis         (EAD vs sex, cardiovascular load, asthma)
        |
        v
08  Single-molecule analysis   (ELOVL2 single-read age inference)
        |
        v
09  Apply clock                (apply trained clock to new BAM/PAT/beta inputs)
```


## Repository layout

```
LongReadClocks/
├── longreadclock/            Python package (importable)
│   ├── preprocessing.py      CpG coordinate loading, PAT helpers, wgbstools wrappers
│   ├── utils.py              GCS helpers, path utilities
│   ├── qc.py                 ML-score QC, bulk PCA, ELOVL2 validation
│   ├── sorting.py            LLR-based single-pass read classification
│   ├── reference.py          Incremental OLS age-reference construction
│   ├── inference.py          Bernoulli grid age inference
│   ├── validation.py         DMR-level sorting validation (MAD)
│   └── plot_style.py         Shared figure style, color palette, fonts
├── notebooks/                Numbered Jupyter notebooks (run in order)
│   ├── 00_setup_environment.ipynb
│   ├── 01_preprocess_bam2pat.ipynb
│   ├── 01b_dsub_preprocess.ipynb      (dsub parallel version of 01)
│   ├── 02_qc_methylation.ipynb
│   ├── 03_cell_type_reference.ipynb
│   ├── 04_digital_sorting.ipynb
│   ├── 04b_dsub_sorting.ipynb         (dsub parallel version of 04)
│   ├── 05_age_reference.ipynb
│   ├── 06_age_prediction.ipynb
│   ├── 07_phenotype_analysis.ipynb
│   ├── 08_single_molecule.ipynb
│   └── 09_apply_clock.ipynb           (apply trained clock to new data)
├── scripts/
│   ├── submit_dsub_preprocess.py   Cloud batch BAM -> PAT jobs (dsub)
│   └── submit_dsub_sort.py         Cloud batch cell-type sorting jobs (dsub)
├── configs/
│   └── default_config.yaml   All pipeline thresholds in one place
├── setup.py
├── requirements.txt
└── .gitignore
```


## Quick start (All of Us Workbench)

```bash
# 1. Clone this repository into your Workbench home directory
cd /home/jupyter
git clone https://github.com/yinjiewu/LongReadClocks.git
cd LongReadClocks

# 2. Install the package (editable mode so notebook edits take effect immediately)
pip install -e .

# 3. Build wgbstools (required for PAT indexing and PAT -> beta conversion)
git clone https://github.com/nloyfer/wgbs_tools.git ~/wgbs_tools
cd ~/wgbs_tools && make -j4
cd /home/jupyter/LongReadClocks

# 4. Open notebooks in order, starting from 00_setup_environment
jupyter lab notebooks/
```


## Batch naming convention

All data batches follow `{site}_{platform}` lowercase naming:

| Batch key   | Site                | Platform        |
|-------------|---------------------|-----------------|
| `bi_revio`  | Broad Institute     | PacBio Revio    |
| `bi_sequel` | Broad Institute     | PacBio Sequel2e |
| `bcm_revio` | Baylor (BCM)        | PacBio Revio    |
| `bcm_sequel`| Baylor (BCM)        | PacBio Sequel2e |
| `bcm_ont`   | Baylor (BCM)        | Oxford Nanopore |
| `jhu_ont`   | Johns Hopkins (JHU) | Oxford Nanopore |
| `uw_revio`  | Univ. Washington    | PacBio Revio    |
| `uw_sequel` | Univ. Washington    | PacBio Sequel2e |
| `ha_revio`  | HudsonAlpha (HA)    | PacBio Revio    |


## GCS folder structure

All outputs land under `$WORKSPACE_BUCKET/results/`:

| GCS path | Content |
|----------|---------|
| `results/{batch_key}/` | Bulk PAT and beta files per batch |
| `results/cell_sorted/{CellType}/` | Sorted PAT and beta per cell type |
| `results/markers/markers_S2/` | Marker BED files from notebook 03 |
| `results/age_references_v2/` | Compact reference bundles (.npy + scalars.json) |
| `results/age_references_v2/reference_index.csv` | Registry of all saved references |
| `results/clock_application/` | Output of 09_apply_clock (CSVs, PNG) |

Cell type names: `Myeloid`, `Lymphoid`, `T_Cell`, `B_Cell`, `NK_Cell`, `Monocyte`, `Granulocyte`.


## Key parameters

All configurable in `configs/default_config.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sorting.tau` | 1.053 | LLR threshold for read assignment |
| `sorting.min_hits` | 5 | Minimum informative CpGs per read |
| `reference.min_samples_per_site` | 20 | Minimum observations per CpG |
| `reference.absr_min` | 0.30 | Minimum absolute R for eligible sites |
| `inference.top_k` | 1000 | Age-informative sites for prediction |
| `inference.depth_cap` | 20 | Read depth ceiling per site |
| `inference.age_grid_step` | 0.1 yr | Discrete age grid resolution |
| `qc.pca_seed` | 20260426 | Reproducibility seed for PCA site sampling |


## Applying the clock to new data

Notebook `09_apply_clock.ipynb` is a self-contained application pipeline. Edit the configuration cell at the top to point at your inputs:

```python
INPUT_FORMAT  = 'BAM'        # 'BAM', 'PAT', or 'BETA'
INPUT_PATHS   = []            # explicit GCS paths, or leave empty to use INPUT_PREFIX
INPUT_PREFIX  = 'gs://fc-secure-xxxx/my_bams/'
CLOCK_TYPES   = ['bulk']     # or a list of cell types, or 'all'
REFERENCE_GROUP_LEVEL = 'technology'
REFERENCE_GROUP_NAME  = 'ONT'
K_INFER       = 1000
RUN_SORTING   = True         # set False if INPUT_FORMAT is 'BETA'
```

The notebook handles the full conversion chain automatically:
`BAM -> PAT -> digital sorting -> cell-type beta -> age prediction`


## Citation

A manuscript is currently in preparation. Citation information will be added upon publication.


## License

MIT License. See `LICENSE` for details.
