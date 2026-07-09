# Real Data Analysis: Human Brain MERFISH

Analysis of the human brain MERFISH aging dataset (Jeffries et al., *Nature*
2025) using stGP and baseline methods. The dataset combines an elderly adult
cohort and an infant adult cohort, profiled with MERFISH on cortical sections.

After QC the dataset contains ~148k cells x 285 genes from 12 sections (9
donors x 1-2 region replicates), spanning 8 ages (15-87 years).

The preprocessing, stGP, and baseline drivers mirror the
[mouse-brain MERFISH pipeline](../RealData_MouseBrainMERFISH/README.md) so
results stay directly comparable. Human-brain figure assembly is currently
notebook-based (`HumanBrain_ext.ipynb`, `HumanBrain_oli.ipynb`) with shared
helpers in `utils.py`, `plots.py`, and `benchmarking_ext.py`.

## Pipeline overview

```
Raw .h5ad ──> 01_preprocess_qc.py ──> QC .h5ad + per-celltype .h5ad files
                                              │
                              ┌───────────────┼─────────────────────────────┐
                              │               │                             │
                       02_run_stgp.py   03a_run_popari.py / 03b_run_spatialpca.R /
                                        03c_run_stamp.py  / 03d_run_mefisto.py
                              │                             │
                         stGP results              Baseline results
                                  ╲                   ╱
                         HumanBrain_ext.ipynb / HumanBrain_oli.ipynb
                                          │
                                  Publication figures
```

## Directory layout

```
RealData_HumanBrainMERFISH/
├── 01_preprocess_qc.py             # QC + per-celltype extraction
├── 02_run_stgp.py                  # Run stGP (uses ../stgp package)
├── 03a_run_popari.py               # Popari baseline (env: Popari)
├── 03b_run_spatialpca.R            # SpatialPCA baseline (env: SpatialPCA_R)
├── 03c_run_stamp.py                # STAMP baseline   (env: stGP / sctm)
├── 03d_run_mefisto.py              # MEFISTO baseline (env: stGP / mofapy2)
├── utils.py                        # Pipeline constants + analysis helpers
├── plots.py                        # Plotting style + figure utilities
├── benchmarking_ext.py             # Consolidated ext benchmark figure helpers
├── Fig3_ext_main_analysis.ipynb     # Main Fig. 3 upstream runbook
├── Fig4_oli_main_analysis.ipynb     # Main Fig. 4 upstream runbook
├── HumanBrain_ext.ipynb            # Excitatory-neuron analysis and figures
├── HumanBrain_oli.ipynb            # Oligodendrocyte analysis and figures
├── downstream_oli/                 # Fig. 4 NicheScope / MCN source-data workflow
└── README.md
```

## Quick start

```bash
# Step 01: QC + per-celltype extraction.
#   * Reads both elderly and infant cohorts under STGP_HUMAN_RAW_DIR
#   * Excludes flagged samples / ages and applies neuronal-contamination +
#     spatial-proximity filters that are specific to the human MERFISH dataset.
#   * Writes data/qc/human_merfish_qc.h5ad and one
#     data/processed/<celltype>.h5ad per cell type.
python3 01_preprocess_qc.py \
    --data-dir "${STGP_HUMAN_RAW_DIR:-data/raw/HumanBrainMERFISH}" \
    --output data/qc/human_merfish_qc.h5ad

# Step 02: Fit stGP for one or every cell type.
#   * Slice grouping is `id_region` (donor_id + region_id), the human analogue
#     of the mouse pipeline's `mouse_id`.
python3 02_run_stgp.py --celltypes micro
python3 02_run_stgp.py --all

# Step 03: Baselines (each script supports --all / --celltypes / single mode).
#   * Run each in its own conda env (see column `env` below).
python3 03a_run_popari.py     --celltypes micro opc          # env: Popari
Rscript  03b_run_spatialpca.R --celltypes micro opc          # env: SpatialPCA_R
python3 03c_run_stamp.py      --celltypes micro opc          # env: stGP (sctm)
python3 03d_run_mefisto.py    --celltypes micro opc          # env: stGP (mofapy2)

# Or one baseline across every cell type:
python3 03a_run_popari.py --all

# Manuscript ext/oli notebooks use the cached STAMP k=3 path:
python3 03c_run_stamp.py --celltypes ext oli --k 3 --out-dir Results/baselines/stamp_k=3

# Main-figure upstream runbooks:
#   Fig3_ext_main_analysis.ipynb, Fig4_oli_main_analysis.ipynb
```

### Main Fig. 3 and Fig. 4 reproduction

- Fig. 3 uses the `ext` stGP result and human benchmark source tables generated
  by `Fig3_ext_main_analysis.ipynb`, `HumanBrain_ext.ipynb`, and
  `benchmarking_ext.py`.
- Fig. 4 uses the `oli` stGP result and NicheScope/MCN source tables documented
  in `Fig4_oli_main_analysis.ipynb` and `downstream_oli/README.md`.
- The final plotting layer is in `../FigureReproducing/`.

Both cached human main-figure stGP fits use automatic program selection. See
`../STGP_PROGRAM_SELECTION.md` for the selected program counts and rationale.

Standalone enrichment templates and working-copy notebooks are not part of the
main-text Fig. 3/Fig. 4 reproduction path. Enrichment or downstream analyses
needed for the main figures are documented in the runbook notebooks above.

### Conda environments

| Script               | Env             | Notes                                 |
| -------------------- | --------------- | ------------------------------------- |
| 01, 02               | `stGP`          | scanpy + the local `stgp/` package    |
| 03a (Popari)         | `Popari`        | needs torch + Popari                  |
| 03b (SpatialPCA)     | `SpatialPCA_R`  | R env with `SpatialPCA`, `Seurat`, `anndataR` |
| 03c (STAMP)          | `stGP` or any env with `sctm`, `squidpy`, `torch` |
| 03d (MEFISTO)        | `stGP` or any env with `mofapy2`, `mofax` |

## Data requirements

The raw per-sample h5ads are not stored in the repo. Set
`STGP_HUMAN_RAW_DIR=/path/to/HumanBrainMERFISH+sc_Nature2025_Jeffries`, or place
the files under `data/raw/HumanBrainMERFISH`. The historical local path was
`/import/home2/share/byual/HumanBrainMERFISH+sc_Nature2025_Jeffries`, but public
reproduction should use a documented data download or archive bundle.

Step 01 expects two sub-directories under `MERFISH_human_aging/`:

- `MERFISH_elderly_adult/MERFISH_h5ad/*.h5ad` (10 sections from 5 donors,
  ages 27-87, sample `5823_rep1` is excluded for batch effects).
- `MERFISH_infant_adult/MERFISH_h5ad/*.h5ad` (4 sections, ages 0.4-57; the
  0.4-year donor is excluded as too young).

After QC the merged AnnData has the columns required by every downstream
step: `donor_id`, `region_id`, `id_region` (= `donor_id + "_" + region_id`),
`age`, `sex`, `cohort`, `celltype`, `celltype2`, `neuronal_contamination_frac`,
plus `obsm["spatial"]`.

## Outputs

- `Results/stgp/<celltype>/` -- `stgp_result.pkl`, `W.csv`,
  `W_active_genes.csv`, and `adata_with_scores.h5ad` (factor scores in
  `obsm["X_stgp"]`, spatial residuals in `obsm["X_stgp_spatial"]`).
- `Results/baselines/<method>/<celltype>/` -- baseline outputs
  (`adata_with_scores.h5ad` with method-specific `obsm` key + `timing.json`;
  Popari stores its AnnData as `res_popari.h5ad`).
- `Results/benchmark_runtimes.jsonl` -- one JSONL line per (method, celltype)
  with wall time and status.
- `Figure/<celltype>/` -- notebook-generated publication figures and source
  data for analyzed human-brain cell types.

## Key differences from the mouse pipeline

These reflect properties of the human dataset and are intentional; the rest
of the pipeline structure is identical so analyses stay directly comparable.

| Aspect | Mouse pipeline | Human pipeline |
| ------ | -------------- | -------------- |
| Grouping column | `mouse_id` | `id_region` (donor_id + region_id) |
| Age unit | months (3-30) | years (15-87) |
| Cell types analysed | 14 (Microglia, T cell, NSC, ...) | 7 (`micro`, `endo`, `opc`, `ast`, `oli`, `inb`, `ext`) |
| Anatomical region | `region` (CTX / STR / CC / VEN) | not annotated -- variance partition uses `celltype2` (subtype) as the region proxy and falls back to age-only OLS |
| QC filters | per-cell counts / genes / bbox area + curated marker exclusions (Sun et al.) | per-cell counts / genes / bbox area / **anisotropy** + **neuronal contamination** + **spatial-proximity-to-neuron** filters specific to the human MERFISH dataset |
| Figure assembly | scripted figure and downstream CLIs | notebook runbooks (`Fig3_ext_main_analysis.ipynb`, `Fig4_oli_main_analysis.ipynb`) plus reusable helpers |

The human-specific QC details (anisotropy, neuronal contamination, neuron
proximity) are implemented in `01_preprocess_qc.py`.
