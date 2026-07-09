"""Step 03d: Run the MEFISTO baseline on per-celltype AnnData files.

Examples:
    python 03d_run_mefisto.py --celltypes Microglia OPC
    python 03d_run_mefisto.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

DEFAULT_OUT_ROOT = Path("Results/baselines/mefisto_new")
CovariateMode = Literal["age", "age_xy"]
SpatialScale = Literal["global", "within_slice"]


def _minmax01(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values.")
    lo = np.min(x)
    hi = np.max(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"{name} has zero or invalid range.")
    return (x - lo) / (hi - lo)


def _scale_xy(
    xy: np.ndarray,
    groups: pd.Series,
    *,
    spatial_scale: SpatialScale,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("Expected spatial coordinates with at least two columns.")
    if not np.isfinite(xy[:, :2]).all():
        raise ValueError("Spatial coordinates contain non-finite values.")

    x = xy[:, 0]
    y = xy[:, 1]
    if spatial_scale == "global":
        return np.column_stack([_minmax01(x, "global x"), _minmax01(y, "global y")])

    if spatial_scale == "within_slice":
        out = np.zeros((xy.shape[0], 2), dtype=float)
        group_values = groups.astype(str).to_numpy()
        for g in pd.unique(group_values):
            idx = np.flatnonzero(group_values == g)
            out[idx, 0] = _minmax01(x[idx], f"x in group {g}")
            out[idx, 1] = _minmax01(y[idx], f"y in group {g}")
        return out

    raise ValueError(f"Unknown spatial_scale: {spatial_scale}")


def _check_one_age_per_group(
    adata,
    *,
    group_col: str,
    age_col: str,
) -> np.ndarray:
    age = pd.to_numeric(adata.obs[age_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(age).all():
        raise ValueError(f"obs['{age_col}'] contains non-numeric or non-finite values.")

    tmp = pd.DataFrame({
        "group": adata.obs[group_col].astype(str).to_numpy(),
        "age": age,
    })
    n_age = tmp.groupby("group")["age"].nunique()
    bad = n_age[n_age != 1]
    if len(bad) > 0:
        raise ValueError(
            f"Expected each {group_col!r} group to have exactly one age. "
            f"Bad groups: {bad.index.tolist()[:10]}"
        )
    if tmp["age"].nunique() < 2:
        raise ValueError("Need at least two distinct ages for temporal MEFISTO.")
    return age


def _extract_factors_and_weights(model_path: Path, obs_names: pd.Index):
    m = mofax.mofa_model(str(model_path))
    factors = m.get_factors(df=True)
    weights = m.get_weights(df=True)
    m.close()

    if isinstance(factors, dict):
        factors = factors.get("mean", factors)
    if isinstance(factors, pd.DataFrame):
        if {"sample", "factor", "value"}.issubset(factors.columns):
            factors = factors.pivot_table(
                index="sample",
                columns="factor",
                values="value",
                aggfunc="first",
            )
        else:
            factors = factors.copy()
    else:
        factors = pd.DataFrame(factors)

    obs_names = pd.Index(obs_names.astype(str))
    if set(obs_names).issubset(set(map(str, factors.index))):
        factors.index = factors.index.astype(str)
        factors = factors.loc[obs_names]
    elif factors.shape[0] == len(obs_names):
        factors.index = obs_names
    else:
        raise ValueError(
            f"MEFISTO returned {factors.shape[0]} factor rows, "
            f"but AnnData has {len(obs_names)} cells."
        )
    return factors, weights


def _write_factor_covariate_diagnostic(
    factors: np.ndarray, cov: np.ndarray, out_dir: Path,
    *, cov_names=("age", "x", "y"),
) -> None:
    """Save a compact diagnostic of factor association with MEFISTO covariates."""
    if factors.size == 0 or cov.size == 0:
        return
    f = np.asarray(factors, dtype=float)
    c = np.asarray(cov, dtype=float)
    f = (f - f.mean(axis=0, keepdims=True)) / (f.std(axis=0, ddof=1, keepdims=True) + 1e-12)
    c = (c - c.mean(axis=0, keepdims=True)) / (c.std(axis=0, ddof=1, keepdims=True) + 1e-12)
    corr = (f.T @ c) / max(f.shape[0] - 1, 1)
    factor_names = [f"MEFISTO{i + 1}" for i in range(corr.shape[0])]
    corr_df = pd.DataFrame(corr, index=factor_names, columns=list(cov_names))
    corr_df.to_csv(out_dir / "factor_covariate_correlations.csv")

    fig, ax = plt.subplots(figsize=(4.2, max(3.2, 0.42 * len(factor_names) + 1.4)),
                           constrained_layout=True)
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            v = corr_df.values[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(v) > 0.55 else "black")
    ax.set_xticks(range(corr_df.shape[1]))
    ax.set_xticklabels(corr_df.columns)
    ax.set_yticks(range(corr_df.shape[0]))
    ax.set_yticklabels(corr_df.index)
    ax.set_xlabel("MEFISTO covariate")
    ax.set_ylabel("Factor")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Pearson r")
    fig.savefig(out_dir / "factor_covariate_correlations.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def run_mefisto(
    input_h5ad: Path, out_dir: Path,
    *,
    n_factors: int,
    n_inducing: int,
    train_iters: int,
    seed: int,
    group_col: str = "mouse_id",
    age_col: str = "age",
    spatial_key: str = "spatial",
    covariate_mode: CovariateMode = "age_xy",
    spatial_scale: SpatialScale = "global",
    target_sum: float = 250.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(str(input_h5ad)).copy()
    adata.obs_names_make_unique()
    if group_col not in adata.obs:
        raise ValueError(f"Expected obs['{group_col}'] to define slice/mouse groups.")
    if age_col not in adata.obs:
        raise ValueError(f"Expected obs['{age_col}'].")
    adata.obs[group_col] = adata.obs[group_col].astype(str)
    adata = adata[adata.obs.sort_values(group_col, kind="stable").index].copy()

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    age = _check_one_age_per_group(adata, group_col=group_col, age_col=age_col)
    cov_df = pd.DataFrame(index=adata.obs_names)
    cov_df["age"] = _minmax01(age, "age")
    cov_cols = ["age"]
    cov_names = ["age"]

    if covariate_mode == "age_xy":
        if spatial_key not in adata.obsm:
            raise ValueError(f"Expected adata.obsm['{spatial_key}'].")
        xy01 = _scale_xy(
            np.asarray(adata.obsm[spatial_key])[:, :2],
            adata.obs[group_col],
            spatial_scale=spatial_scale,
        )
        cov_df["x"] = xy01[:, 0]
        cov_df["y"] = xy01[:, 1]
        cov_cols += ["x", "y"]
        cov_names += ["x", "y"]
    elif covariate_mode != "age":
        raise ValueError(f"Unknown covariate_mode: {covariate_mode}")

    for c in cov_cols:
        adata.obs[f"mefisto_{c}01"] = cov_df[c].to_numpy(dtype=float)

    frac_inducing = float(min(max(n_inducing / max(1, adata.n_obs), 1e-4), 0.8))

    ent = entry_point()
    ent.set_data_options(use_float32=True, center_groups=False)
    ent.set_data_from_anndata(adata, groups_label=group_col, save_metadata=True)
    ent.set_model_options(factors=int(n_factors))
    ent.set_train_options(iter=int(train_iters), convergence_mode="fast",
                          seed=int(seed), verbose=False, quiet=True)
    cov_list = [
        cov_df.loc[samples, cov_cols].to_numpy(dtype=float)
        for samples in ent.data_opts["samples_names"]
    ]
    ent.set_covariates(cov_list, covariates_names=cov_names)
    ent.set_smooth_options(
        scale_cov=False,
        sparseGP=True,
        frac_inducing=frac_inducing,
        start_opt=10,
        opt_freq=10,
        model_groups=False,
    )
    ent.build()
    ent.run()

    model_path = out_dir / "model.hdf5"
    ent.save(str(model_path), save_data=False, expectations=["W", "Z"])

    factors, weights = _extract_factors_and_weights(model_path, adata.obs_names)

    adata.obsm["X_mefisto"] = factors.to_numpy(dtype=np.float32)
    adata.uns["mefisto"] = dict(
        n_factors=int(n_factors), n_inducing=int(n_inducing),
        frac_inducing=frac_inducing, train_iters=int(train_iters), seed=int(seed),
        model="MEFISTO_Yt_grouped_covariate_GP",
        group_col=group_col, age_col=age_col, spatial_key=spatial_key,
        covariate_mode=covariate_mode, spatial_scale=spatial_scale,
        covariates=cov_names, center_groups=False, model_groups=False,
        scale_cov=False, sparseGP=True, target_sum=float(target_sum),
    )
    adata.write_h5ad(out_dir / "adata_with_scores.h5ad", compression="gzip")
    factors.to_csv(out_dir / "factors.csv")
    weights.to_csv(out_dir / "weights.csv")
    cov_df.to_csv(out_dir / "covariates_used_by_mefisto.csv")
    (out_dir / "run_config.json").write_text(json.dumps(adata.uns["mefisto"], indent=2))
    _write_factor_covariate_diagnostic(
        adata.obsm["X_mefisto"],
        cov_df[cov_cols].to_numpy(dtype=float),
        out_dir,
        cov_names=cov_names,
    )
    print(f"[MEFISTO] done. wrote: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the MEFISTO baseline on per-celltype MERFISH AnnData.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--all", action="store_true", help="Run every cell type.")
    ap.add_argument("--celltypes", nargs="+", metavar="CT",
                    help="Run only these cell types.")
    ap.add_argument("--data-dir", type=Path, default=DATA_PROCESSED,
                    help="Directory with per-celltype .h5ad files from step 01.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--k", type=int, default=4, help="Number of factors.")
    ap.add_argument("--n-inducing", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-col", default="mouse_id")
    ap.add_argument("--age-col", default="age")
    ap.add_argument("--spatial-key", default="spatial")
    ap.add_argument("--covariate-mode", choices=["age", "age_xy"], default="age_xy")
    ap.add_argument("--spatial-scale", choices=["global", "within_slice"], default="global")
    ap.add_argument("--target-sum", type=float, default=250.0)
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
                group_col=args.group_col, age_col=args.age_col,
                spatial_key=args.spatial_key, covariate_mode=args.covariate_mode,
                spatial_scale=args.spatial_scale, target_sum=args.target_sum,
            ),
            out_dir=out_dir, timing_log=TIMING_LOG,
            extra_timing={"k": args.k, "n_inducing": args.n_inducing,
                          "train_iters": args.iters, "seed": args.seed,
                          "group_col": args.group_col,
                          "covariate_mode": args.covariate_mode,
                          "spatial_scale": args.spatial_scale},
        )


if __name__ == "__main__":
    main()
