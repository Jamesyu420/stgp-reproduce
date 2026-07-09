"""Step 03a: Run the Popari baseline (requires the ``Popari`` conda env).

Examples:
    python 03a_run_popari.py --celltypes Microglia
    python 03a_run_popari.py --celltypes Microglia OPC --device cuda:1
    python 03a_run_popari.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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


def run_popari(
    input_h5ad: Path, out_dir: Path,
    *, k_topics: int, device: str, nmf_iters: int, train_iters: int, seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5ad = out_dir / "res_popari.h5ad"

    torch.manual_seed(seed)
    np.random.seed(seed)

    adata = sc.read_h5ad(str(input_h5ad))

    # Per-mouse PopariDataset, normalised + log1p, with spatial neighbours.
    popari_datasets: list[PopariDataset] = []
    for g in adata.obs["mouse_id"].astype(str).unique():
        ds = adata[adata.obs["mouse_id"].astype(str) == g].copy()
        age_val = ds.obs["age"].iloc[0]
        sc.pp.normalize_total(ds, target_sum=250.0, inplace=True)
        sc.pp.log1p(ds)
        pop_ds = PopariDataset(ds, f"{g}_age{age_val:g}")
        pop_ds.compute_spatial_neighbors()
        popari_datasets.append(pop_ds)

    dataset_path = out_dir / "preprocessed_popari_dataset.h5ad"
    save_anndata(str(dataset_path), popari_datasets)

    model = Popari(
        K=k_topics, dataset_path=str(dataset_path),
        lambda_Sigma_x_inv=1e-4, lambda_Sigma_bar=1e-4,
        initial_context=dict(device=device, dtype=torch.float64),
        torch_context=dict(device=device, dtype=torch.float64),
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
        description="Run the Popari baseline on per-celltype MERFISH AnnData "
                    "(requires conda env 'Popari')."
    )
    ap.add_argument("--all", action="store_true", help="Run every cell type.")
    ap.add_argument("--celltypes", nargs="+", metavar="CT",
                    help="Run only these cell types.")
    ap.add_argument("--data-dir", type=Path, default=DATA_PROCESSED,
                    help="Directory with per-celltype .h5ad files from step 01.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--k", type=int, default=4, help="Number of topics.")
    ap.add_argument("--device", type=str, default="cuda:2")
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
                i, o, k_topics=args.k, device=args.device,
                nmf_iters=args.nmf_iters, train_iters=args.train_iters,
                seed=args.seed,
            ),
            out_dir=out_dir, timing_log=TIMING_LOG,
            extra_timing={"k": args.k, "device": args.device,
                          "nmf_iters": args.nmf_iters,
                          "train_iters": args.train_iters, "seed": args.seed},
        )


if __name__ == "__main__":
    main()
