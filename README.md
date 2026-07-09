# Characterizing dynamic tissue architectures by identifying cell-type-specific spatiotemporal gene programs with stGP

This repository contains the reproducing code for the analysis in the **stGP** (spatiotemporal Gene Programs for spatial transcriptomics by Gaussian Process) paper, which develops a statistical framework for discovering spatiotemporally variable gene programs from aging spatial transcriptomic data.

## Introduction

stGP is a statistical framework for identifying interpretable cell-type-specific spatiotemporal gene programs (stGPs) from multi-sample spatiotemporal transcriptomic data measured across biological time.

stGP's effectiveness relies on our innovations in the integration of Gaussian process priors and interpretable matrix factorization:

- stGP represents gene expression within each cell type as a small set of latent programs with non-negative gene loadings constrained to the simplex, making each program interpretable as a weighted gene set shared across samples.
- stGP decomposes per-cell program activity into a sample-level temporal component that captures coordinated responses over biological time (e.g., age or stage), and a within-section spatial component that characterizes local program deployment across tissue coordinates—without requiring cross-section registration.
- Gaussian process priors smooth temporal trajectories along the biological-time covariate and spatial embeddings within each tissue section, while variance components quantify the relative contributions of time and space to each program.
- For multi-program inference, stGP adopts a blockwise backfitting scheme that sequentially extracts rank-1 components from residuals, with automatic model selection to determine the number of programs.

<p align="center">
  <img src="FigureReproducing/Fig1_overview.png" width="85%" alt="Overview" />
</p>



## Installation

```bash
# Clone or copy the repository
cd stGP
conda env create -f stGP.yml
conda activate stGP
pip install -e .
```

## Reference

If you find `stGP` or any of the source code in this repository and https://github.com/YangLabHKUST/stGP useful for your work, please cite:
> Characterizing dynamic tissue architectures by identifying cell-type-specific spatiotemporal gene programs with stGP.
> Baichen Yu, Ziyue Tan, Xiaomeng Wan, Hansheng Wang, and Can Yang.
> Preprint at Biorxiv, 2026.
> https://doi.org/10.64898/2026.07.03.736035

## Development

The software is developed and maintained by [Baichen Yu](mailto:mabyu@ust.hk).

## Contact

Please feel free to contact [Baichen Yu](mailto:mabyu@ust.hk) or [Prof. Can Yang](mailto:macyang@ust.hk) if any inquiries.