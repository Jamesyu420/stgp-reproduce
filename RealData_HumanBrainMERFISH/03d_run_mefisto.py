"""Step 03d: Run the MEFISTO baseline on per-celltype Human Brain MERFISH AnnData.

Examples::

    python 03d_run_mefisto.py --celltypes micro opc
    python 03d_run_mefisto.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import mofax
from mofapy2.run.entry_point import entry_point

from utils import (
    BASELINE_RUN_ORDER,
    DATA_PROCESSED,
    TIMING_LOG,
    safe_name,
    time_and_log_baseline,
)

DEFAULT_OUT_ROOT = Path("Results/baselines/mefisto")


def run_mefisto(
    input_h5ad: Path, out_dir: Path,
    *, n_factors: int, n_inducing: int, train_iters: int, seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(str(input_h5ad))
    sc.pp.normalize_total(adata, target_sum=250.0)
    sc.pp.log1p(adata)

    age = pd.to_numeric(adata.obs["age"], errors="coerce").to_numpy(
        dtype=float).reshape(-1, 1)
    if np.isnan(age).any():
        raise ValueError("Found NaN in obs['age'] after numeric conversion.")
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    cov = np.concatenate([age, xy], axis=1)
    cov = (cov - cov.mean(0)) / cov.std(0, ddof=1)

    frac_inducing = float(min(max(n_inducing / max(1, adata.n_obs), 1e-4), 0.8))

    ent = entry_point()
    ent.set_data_options(use_float32=True)
    ent.set_data_from_anndata(adata)
    ent.set_model_options(factors=int(n_factors))
    ent.set_train_options(iter=int(train_iters), convergence_mode="fast",
                          seed=int(seed), verbose=False, quiet=True)
    ent.set_covariates([cov], covariates_names=["age", "x", "y"])
    ent.set_smooth_options(sparseGP=True, frac_inducing=frac_inducing,
                           start_opt=10, opt_freq=10)
    ent.build()
    ent.run()

    model_path = out_dir / "model.hdf5"
    ent.save(str(model_path), save_data=False, expectations=["W", "Z"])

    m = mofax.mofa_model(str(model_path))
    factors = m.get_factors(df=True)
    weights = m.get_weights(df=True)
    m.close()

    adata.obsm["X_mefisto"] = factors.to_numpy(dtype=np.float32)
    adata.uns["mefisto"] = dict(
        n_factors=int(n_factors), n_inducing=int(n_inducing),
        frac_inducing=frac_inducing, train_iters=int(train_iters), seed=int(seed),
    )
    adata.write_h5ad(out_dir / "adata_with_scores.h5ad", compression="gzip")
    weights.T.to_csv(out_dir / "weights.csv")
    print(f"[MEFISTO] done. wrote: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the MEFISTO baseline on per-celltype Human Brain MERFISH AnnData.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--all", action="store_true", help="Run every cell type.")
    ap.add_argument("--celltypes", nargs="+", metavar="CT",
                    help="Run only these cell types.")
    ap.add_argument("--data-dir", type=Path, default=DATA_PROCESSED)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--k", type=int, default=4, help="Number of factors.")
    ap.add_argument("--n-inducing", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=1000)
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
        print(f"\n{'=' * 60}\n[MEFISTO] {ct}\n{'=' * 60}")
        if not input_h5ad.exists():
            print(f"[error] missing {input_h5ad}")
            continue
        if (out_dir / "adata_with_scores.h5ad").exists():
            print(f"[skip] {out_dir} already done.")
            continue
        time_and_log_baseline(
            method="mefisto", celltype=ct,
            fn=lambda i=input_h5ad, o=out_dir: run_mefisto(
                i, o, n_factors=args.k, n_inducing=args.n_inducing,
                train_iters=args.iters, seed=args.seed,
            ),
            out_dir=out_dir, timing_log=TIMING_LOG,
            extra_timing={"k": args.k, "n_inducing": args.n_inducing,
                          "train_iters": args.iters, "seed": args.seed},
        )


if __name__ == "__main__":
    main()
