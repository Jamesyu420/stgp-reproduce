"""Step 03a: Run the Popari baseline on per-celltype Human Brain MERFISH AnnData.

Requires the ``Popari`` conda env.

Examples::

    python 03a_run_popari.py --celltypes micro
    python 03a_run_popari.py --celltypes micro opc --device cuda:1
    python 03a_run_popari.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from popari.components import PopariDataset
from popari.io import save_anndata
from popari.model import Popari

from utils import (
    BASELINE_RUN_ORDER,
    DATA_PROCESSED,
    TIMING_LOG,
    safe_name,
    time_and_log_baseline,
)

DEFAULT_OUT_ROOT = Path("Results/baselines/popari")


def _normalize_log1p(adata: ad.AnnData, *, target_sum: float = 250.0) -> None:
    sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)
    sc.pp.log1p(adata)


def _build_popari_datasets(
    adata: ad.AnnData,
    *,
    n_hvg: int = 200,
    target_sum: float = 250.0,
) -> list[PopariDataset]:
    """Per-sample ``PopariDataset`` list with shared HVG selection."""
    groups = adata.obs["id_region"].astype(str).unique().tolist()
    datasets, dataset_names = [], []
    for g in groups:
        ds = adata[adata.obs["id_region"].astype(str) == g].copy()
        if ds.n_obs == 0:
            continue
        age_val = (
            float(pd.to_numeric(ds.obs["age"], errors="coerce").dropna().iloc[0])
            if "age" in ds.obs else float("nan")
        )
        datasets.append(ds)
        dataset_names.append(f"{g}_age{age_val:g}")
    if len(datasets) < 2:
        raise ValueError(f"Need >=2 samples for Popari; got {len(datasets)}.")

    merged = ad.concat(datasets, label="batch", keys=dataset_names,
                       merge="unique", uns_merge="unique")
    _normalize_log1p(merged, target_sum=target_sum)
    n_hvg_eff = int(min(max(1, n_hvg), merged.n_vars))
    sc.pp.highly_variable_genes(merged, n_top_genes=n_hvg_eff)
    hv_mask = (merged.var["highly_variable"].to_numpy()
               if "highly_variable" in merged.var
               else np.ones(merged.n_vars, dtype=bool))
    hv_genes = merged.var_names[hv_mask]

    popari_datasets: list[PopariDataset] = []
    for ds, name in zip(datasets, dataset_names):
        _normalize_log1p(ds, target_sum=target_sum)
        ds = ds[:, hv_genes].copy()
        pop_ds = PopariDataset(ds, name)
        pop_ds.compute_spatial_neighbors()
        popari_datasets.append(pop_ds)
    return popari_datasets


def run_popari(
    input_h5ad: Path, out_dir: Path,
    *, k_topics: int, n_hvg: int, device: str, dtype: str,
    nmf_iters: int, train_iters: int, seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5ad = out_dir / "res_popari.h5ad"

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_dtype = torch.float64 if dtype == "float64" else torch.float32

    adata = sc.read_h5ad(str(input_h5ad))
    if "id_region" not in adata.obs:
        raise ValueError("Expected obs['id_region'] in input AnnData.")
    if "spatial" not in adata.obsm:
        raise ValueError("Expected obsm['spatial'] in input AnnData.")

    popari_datasets = _build_popari_datasets(
        adata, n_hvg=n_hvg, target_sum=250.0,
    )

    dataset_path = out_dir / "preprocessed_popari_dataset.h5ad"
    save_anndata(str(dataset_path), popari_datasets)

    model = Popari(
        K=k_topics, dataset_path=str(dataset_path),
        lambda_Sigma_x_inv=1e-4, lambda_Sigma_bar=1e-4,
        initial_context=dict(device=device, dtype=torch_dtype),
        torch_context=dict(device=device, dtype=torch_dtype),
        verbose=0, spatial_affinity_mode="differential lookup",
    )

    # NMF-style warm-up (no spatial-affinity update yet).
    for _ in range(nmf_iters):
        model.estimate_parameters(update_spatial_affinities=False)
        model.estimate_weights(use_neighbors=False)

    # Main training loop with spatial affinities + neighbour-based weights.
    for _ in range(train_iters):
        model.estimate_parameters()
        model.estimate_weights()

    model.save_results(str(out_dir / "res_popari"), ignore_raw_data=False)
    print(f"[Popari] done. wrote: {out_h5ad}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the Popari baseline on per-celltype Human Brain MERFISH "
                    "AnnData (requires conda env 'Popari').",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--all", action="store_true", help="Run every cell type.")
    ap.add_argument("--celltypes", nargs="+", metavar="CT",
                    help="Run only these cell types.")
    ap.add_argument("--data-dir", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--k", type=int, default=4, help="Number of topics.")
    ap.add_argument("--n-hvg", type=int, default=200)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, choices=["float32", "float64"],
                    default="float64")
    ap.add_argument("--nmf-iters", type=int, default=10)
    ap.add_argument("--train-iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.all:
        celltypes = BASELINE_RUN_ORDER
    elif args.celltypes:
        celltypes = args.celltypes
    else:
        raise SystemExit("Pick cell types with --all or --celltypes CT1 CT2 ...")

    for ct in celltypes:
        safe_ct = safe_name(ct)
        input_h5ad = args.data_dir / f"{safe_ct}.h5ad"
        out_dir = args.out_dir / safe_ct
        print(f"\n{'=' * 60}\n[Popari] {ct}\n{'=' * 60}")
        if not input_h5ad.exists():
            print(f"[error] missing {input_h5ad}")
            continue
        if (out_dir / "res_popari.h5ad").exists():
            print(f"[skip] {out_dir} already done.")
            continue
        time_and_log_baseline(
            method="popari", celltype=ct,
            fn=lambda i=input_h5ad, o=out_dir: run_popari(
                i, o, k_topics=args.k, n_hvg=args.n_hvg,
                device=args.device, dtype=args.dtype,
                nmf_iters=args.nmf_iters, train_iters=args.train_iters,
                seed=args.seed,
            ),
            out_dir=out_dir, timing_log=TIMING_LOG,
            extra_timing={"k": args.k, "n_hvg": args.n_hvg, "device": args.device,
                          "nmf_iters": args.nmf_iters,
                          "train_iters": args.train_iters, "seed": args.seed},
        )


if __name__ == "__main__":
    main()
