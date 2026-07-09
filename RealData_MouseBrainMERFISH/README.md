# Real Data Analysis: Mouse Brain MERFISH

Analysis of the aging mouse brain MERFISH spatial transcriptomics dataset using
stGP and baseline methods.  The dataset contains approximately one million cells
across multiple age groups (3--30 months), profiled with MERFISH on coronal
brain sections.

## Pipeline overview

```
Raw .h5ad ──> 01_preprocess_qc.py ──> QC .h5ad + per-celltype .h5ad files
                                              │
                              ┌───────────────┼─────────────────────────────┐
                              │               │                             │
              MouseBrain_microglia.ipynb  03a_run_popari.py / 03b_run_spatialpca.R /
                                        03c_run_stamp.py / 03d_run_mefisto.py
                              │                             │
                         stGP results              Baseline results
                                  ╲                   ╱
                              MouseBrain_microglia.ipynb
                                          │
                                  Publication figures
                                          │
                              MouseBrain_downstream.ipynb
```

## Directory layout

```
RealData_MouseBrainMERFISH/
├── 01_preprocess_qc.py             # QC + per-celltype extraction
├── 03a_run_popari.py               # Popari baseline
├── 03b_run_spatialpca.R            # SpatialPCA baseline
├── 03c_run_stamp.py                # STAMP baseline
├── 03d_run_mefisto.py              # MEFISTO baseline
├── MouseBrain_microglia.ipynb      # stGP fit + publication/source figures
├── MouseBrain_downstream.ipynb     # proximity + pathway/signature enrichment
├── Fig5_Microglia_reproduction.md  # Main Fig. 5 tutorial wrapper
├── downstream_cluster_Microglia.ipynb # Slingshot / clustering downstream notebook
├── utils.py                        # Shared paths/I/O/statistical helpers
└── plots.py                        # Consolidated plotting/loading helpers
```

## Quick start

```bash
# Step 01: QC + extract per-celltype files (writes data/processed/<celltype>.h5ad)
python3 01_preprocess_qc.py \
    --input data/raw/aging_coronal.h5ad \
    --output data/qc/aging_coronal_qc.h5ad

# Step 02-style tutorial: fit/reuse stGP and generate figures
jupyter notebook MouseBrain_microglia.ipynb

# Baselines (each script supports --all / --celltypes / single mode):
python3 03a_run_popari.py     --celltypes Microglia OPC          # conda env: Popari
Rscript  03b_run_spatialpca.R --celltypes Microglia OPC          # conda env: SpatialPCA_R
python3 03c_run_stamp.py      --celltypes Microglia OPC          # conda env: stGP / sctm
python3 03d_run_mefisto.py    --celltypes Microglia OPC          # conda env: stGP / mofapy2

# Or all cell types for a single baseline:
python3 03a_run_popari.py --all

# Downstream proximity and enrichment tutorial:
jupyter notebook MouseBrain_downstream.ipynb
```

## Data requirements

Place the raw MERFISH dataset at `data/raw/aging_coronal.h5ad`, or set
`STGP_MOUSE_RAW_H5AD=/path/to/aging_coronal.h5ad`. This file is not included in
the repository due to its size (~4 GB). The QC-filtered version and the
per-cell-type files are created by step 01.

## Main Fig. 5 reproduction

The upstream chain for manuscript Fig. 5 is documented in
`Fig5_Microglia_reproduction.md`, `MouseBrain_microglia.ipynb`, and
`MouseBrain_downstream.ipynb`. The final plotting layer is
`../FigureReproducing/Fig5_mouse_brain.ipynb`.

The cached Microglia result fixes the stGP program count at `p=4` for matched
method benchmarking. This is intentional and is documented in
`../STGP_PROGRAM_SELECTION.md`.

`downstream_cluster_Microglia.ipynb` remains separate because it requires the
NicheScope environment. The tutorial notebooks, `utils.py`, and consolidated
`plots.py` are the authoritative upstream code for Fig. 5.

## Outputs

- `Results/stgp/<celltype>/`:  `stgp_result.pkl`, `W.csv`, and an annotated
  AnnData with factor scores in `obsm["X_stgp"]` and spatial residuals in
  `obsm["X_stgp_spatial"]`.
- `Results/baselines/<method>/<celltype>/`:  Baseline method results
  (`adata_with_scores.h5ad` with method-specific `obsm` key, plus `timing.json`;
  Popari stores its AnnData as `res_popari.h5ad`).
- `Results/benchmark_runtimes.jsonl`:  one JSONL line per (method, celltype)
  with wall time and status.
- `Figures/<celltype>/`:  Publication figures.
- `Results/proximity/<celltype>/` and `Figures/<celltype>/proximity/`:
  spatial-proximity downstream tables and figures.
- `Results/enrichment/<celltype>/` and `Figures/<celltype>/enrichment/`:
  pathway / cell-type signature enrichment tables and figures.
