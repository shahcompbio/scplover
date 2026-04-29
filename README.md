# scPlOver
Method scPlOver for inferring DNA content from overlapping fragments in scDNA-seq data. 

**COMING SOON: end-to-end pipeline to run scPlOver starting from a BAM file (including counting fragment overlaps and running HMMCopy)**

# Requirements (and tested versions) 
* Python (3.9)
* numpy (1.23.0)
* scipy (1.13.1)
* pandas (2.1.4)
* statsmodels (0.14.1)
* anndata (0.10.3)
* click (8.1.7)

# Input
See `test_input/FUCCI.h5ad` for example.

Anndata of cells x bins, with obs indices containing cell IDs, var indices containing bins in `chr:start-end` format, and layers:
* `X` containing read count
* `state` containing integer copy number state
* `overlaps` containing the number of fragment overlaps
* `overlap_bases` containing the number of overlap bases
* `n_fragments` containing the total number of fragments
* `mean_fragment_length` containing the average fragment length

Required var fields:

* `chr`: chromosome
* `start`:  bin start position
* `end`: bin end position
* `gc`: GC content
* `in_blacklist`: flag indicating whether bin is present in blacklist
* `map`: mappability (currently unused)

# Usage
```
python run_scplover_adata.py \
  --adata <path> \
  --output_row <path> \
  --output_table <path> \
  --output_adata <path> \
  --cell_df_dir <dir> \
  [options]
```

## Example usage
Running on test data:
```
python run_scplover_adata.py \
  --adata SPECTRUM-OV-001_hmmcopy_overlaps.h5ad \
  --output_row results/SPECTRUM-OV-001_0_table.csv \
  --output_table results/SPECTRUM-OV-001_0_full_table.csv \
  --output_adata results/SPECTRUM-OV-001_0_adata.h5ad \
  --cell_df_dir results/cell_dfs \
  --max_k 12 \
  --covariance_type full \
  --means scale \
  --bounds 0.8,1.2
  --iqr_threshold 5 \
  --cores 4
```

Typical usage in paper for experimental datasets:
```
python run_scplover_adata.py --adata {input.adata} \
  --output_row {output.row} \
  --output_table {output.full_table} \
  --output_adata {output.adata} \
  --cell_df_dir {params.cell_df_dir} \
  --max_k 12 \
  --cells {params.cells_arg} \
  --iqr_threshold 2 \
  --covariance_type full \
  --correct_gc \
  --bases_dist_quantile 0.8 \
  --lowess_frac 0.2 \
  --min_mean_scale 0.8 \
  --max_mean_scale 1.2 \
  --means scale \
  --fit_transitions \
  --min_bins_per_state 50 \
```

## Required arguments

| Argument | Description |
|---|---|
| `--adata` | Path to input `.h5ad` file containing read counts and overlap bases per bin per cell |
| `--output_row` | Output CSV with one row per cell (best-scoring model result) |
| `--output_table` | Output CSV with all results across all ploidy initializations |
| `--output_adata` | Output `.h5ad` with `ghmm_state` layer added containing inferred copy number states |
| `--cell_df_dir` | Directory to write per-cell regression DataFrames (one CSV per cell) |

## Cell selection (mutually exclusive)

| Argument | Default | Description |
|---|---|---|
| `--cells` | all cells | Comma-separated list of cell IDs to process |
| `--cells_file` | — | File with one cell ID per line to process |

## Model options

| Argument | Default | Description |
|---|---|---|
| `--max_k` | `12` | Maximum copy number state (max supported: 29) |
| `--covariance_type` | `full` | Covariance structure: `full`, `diag`, or `spherical` |
| `--means` | `fixed` | Mean treatment: `fixed` (held at initial values), `scale` (learned per-feature scale factor), or `free` (fully learned) |
| `--fit_transitions` | `False` | If set, learn transition matrix from data (default: fixed) |
| `--include_increments` | `False` | If set, also explore ploidy initializations formed by integer increments from the input copy number states (otherwise, consider multiples only) |

## Mean scaling bounds (used when `--means scale`)

| Argument | Default | Description |
|---|---|---|
| `--min_mean_scale` | `0` | Lower bound on per-feature mean scale factor |
| `--max_mean_scale` | `inf` | Upper bound on per-feature mean scale factor |
| `--scale_reads` | `False` | If set, apply mean scaling to the reads dimension; otherwise only the overlap bases dimension is scaled |

## Filtering options

| Argument | Default | Description |
|---|---|---|
| `--iqr_threshold` | `None` | IQR multiplier for outlier removal per state (e.g. `5`); disabled if not set |
| `--min_bins_per_state` | `0` | Minimum number of bins a state must have; bins in rarer states are removed before fitting |

## GC correction options

| Argument | Default | Description |
|---|---|---|
| `--correct_gc` | `False` | If set, apply LOWESS-based modal quantile GC correction to reads and overlap bases |
| `--lowess_frac` | `0.2` | Fraction of data used in LOWESS smoothing during GC correction |
| `--clip_corrected_values` | `False` | If set, clip GC-corrected values to a valid range |
| `--bases_dist_quantile` | `0.8` | Quantile of the overlap bases distribution used during GC correction |

## Performance

| Argument | Default | Description |
|---|---|---|
| `--cores` | `1` | Number of parallel worker processes for fitting cells |


