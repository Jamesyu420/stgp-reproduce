"""Step 03c: Run the STAMP baseline on per-celltype AnnData files.

Examples:
    python 03c_run_stamp.py --celltypes Microglia OPC --device cuda:0
    python 03c_run_stamp.py --all
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import sctm
import squidpy as sq
import torch

from utils import (
    BASELINE_RUN_ORDER,
    DATA_PROCESSED,
    TIMING_LOG,
    safe_name,
    time_and_log_baseline,
)

DEFAULT_OUT_ROOT = Path("Results/baselines/stamp")

MIN_CUTOFF_SCHEDULE = (0.00, 0.01, 0.05, 0.10)
TOP_K_GENES = 15

def _has_nan(topic_prop, beta) -> bool:
    """True if STAMP outputs contain NaN (silent training failure)."""
    if topic_prop is None or topic_prop.isna().to_numpy().any():
        return True
    if hasattr(beta, "isna"):
        return bool(beta.isna().to_numpy().any())
    if isinstance(beta, dict):
        return any(df.isna().to_numpy().any() for df in beta.values())
    return False


def _top_genes_per_topic(W: pd.DataFrame, n_topics: int, top_k: int) -> pd.DataFrame:
    """Average W (= w_new) over time, return top-k genes per topic.

    W has MultiIndex (time_code, "TopicX") and gene columns; same convention as
    `STAMP.get_feature_by_topic()` for time-series. Higher score = more
    topic-specific after low-expression downweighting.
    """
    mean_per_topic = W.groupby(level=1, sort=False).mean()
    records = []
    for topic in [f"Topic{k}" for k in range(1, n_topics + 1)]:
        if topic not in mean_per_topic.index:
            continue
        weights = mean_per_topic.loc[topic]
        order = weights.sort_values(ascending=False).head(top_k)
        for rank, (gene, score) in enumerate(order.items(), start=1):
            records.append({
                "topic": topic, "rank": rank, "gene": gene,
                "mean_score": float(score),
            })
    return pd.DataFrame(records)


def run_stamp(
    input_h5ad: Path, out_dir: Path,
    *, n_topics: int, device: str, max_epochs: int, min_epochs: int, seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    base = sc.read_h5ad(str(input_h5ad))
    n_neighs = max(5, round(0.006 * base.n_obs))
    sq.gr.spatial_neighbors(base, n_neighs=n_neighs, coord_type="generic")
    base.obs["age"] = base.obs["age"].astype("float32")
    sc.pp.filter_cells(base, min_counts=50)

    # Retry over progressively stricter gene filters until training is stable.
    model = topic_prop = adata = None
    used_cutoff = float("nan")
    last_error: Exception | None = None

    for attempt, min_cutoff in enumerate(MIN_CUTOFF_SCHEDULE, start=1):
        adata_try = base.copy()
        sctm.pp.filter_genes(adata_try, min_cutoff=min_cutoff)
        print(f"[STAMP] attempt {attempt}/{len(MIN_CUTOFF_SCHEDULE)}: "
              f"min_cutoff={min_cutoff:g} -> {adata_try.n_vars} genes kept")
        if adata_try.n_vars == 0:
            continue

        try:
            sctm.seed.seed_everything(seed)
            m = sctm.stamp.STAMP(
                adata_try, n_topics=n_topics, time_covariate_keys="age",
                mode="sgc", gene_likelihood="nb", verbose=False,
            )
            m.train(
                device=device, sampler="W",
                max_epochs=max_epochs, min_epochs=min_epochs,
                learning_rate=0.01, batch_size=256,
                early_stop=True, shuffle=True,
            )
            tp = m.get_cell_by_topic()
            beta = m.get_feature_by_topic()
            if _has_nan(tp, beta):
                raise RuntimeError("STAMP outputs contain NaN after training.")
        except Exception as e:
            last_error = e
            print(f"[STAMP]   failed at min_cutoff={min_cutoff:g}: "
                  f"{type(e).__name__}: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        model, topic_prop, adata = m, tp, adata_try
        used_cutoff = min_cutoff
        print(f"[STAMP] succeeded with min_cutoff={min_cutoff:g} "
              f"({adata.n_vars} genes used)")
        break
    else:
        raise RuntimeError(
            f"STAMP failed for all min_cutoffs ({list(MIN_CUTOFF_SCHEDULE)}). "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )

    W = model.get_feature_by_topic()
    W.to_csv(out_dir / "W_loadings.csv")

    n_time = model.n_time
    n_genes = model.n_features
    W_arr = np.stack(
        [W.loc[t].reindex([f"Topic{k}" for k in range(1, n_topics + 1)]).to_numpy()
         for t in range(n_time)],
        axis=-1,
    )
    assert W_arr.shape == (n_topics, n_genes, n_time), W_arr.shape
    np.save(out_dir / "W_loadings.npy", W_arr)

    top_genes_df = _top_genes_per_topic(W, n_topics=n_topics, top_k=TOP_K_GENES)
    top_genes_df.to_csv(out_dir / "top_genes_per_topic.csv", index=False)

    adata.obsm["X_stamp"] = topic_prop.to_numpy(dtype=np.float32)
    adata.uns["stamp"] = dict(
        n_topics=n_topics, max_epochs=max_epochs, min_epochs=min_epochs,
        n_neighs=n_neighs, seed=seed, min_cutoff=used_cutoff,
        n_genes_used=int(adata.n_vars),
        top_k_genes=TOP_K_GENES,
    )
    adata.write_h5ad(out_dir / "adata_with_scores.h5ad", compression="gzip")
    print(f"[STAMP] done. wrote: {out_dir}")
    print(f"         W_loadings.csv         (time*topic x gene, shape={W.shape})")
    print(f"         W_loadings.npy         (topics x genes x time, shape={W_arr.shape})")
    print(f"         top_genes_per_topic.csv ({n_topics} topics x {TOP_K_GENES} genes)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the STAMP baseline on per-celltype MERFISH AnnData."
    )
    ap.add_argument("--all", action="store_true", help="Run every cell type.")
    ap.add_argument("--celltypes", nargs="+", metavar="CT",
                    help="Run only these cell types.")
    ap.add_argument("--data-dir", type=Path, default=DATA_PROCESSED,
                    help="Directory with per-celltype .h5ad files from step 01.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--k", type=int, default=4, help="Number of topics.")
    ap.add_argument("--device", type=str, default="cuda:4")
    ap.add_argument("--max-epochs", type=int, default=400)
    ap.add_argument("--min-epochs", type=int, default=100)
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
        print(f"\n{'=' * 60}\n[STAMP] {ct}\n{'=' * 60}")
        if not input_h5ad.exists():
            print(f"[error] missing {input_h5ad}")
            continue
        if (out_dir / "adata_with_scores.h5ad").exists():
            print(f"[skip] {out_dir} already done.")
            continue
        time_and_log_baseline(
            method="stamp", celltype=ct,
            fn=lambda i=input_h5ad, o=out_dir: run_stamp(
                i, o, n_topics=args.k, device=args.device,
                max_epochs=args.max_epochs, min_epochs=args.min_epochs,
                seed=args.seed,
            ),
            out_dir=out_dir, timing_log=TIMING_LOG,
            extra_timing={"k": args.k, "device": args.device,
                          "max_epochs": args.max_epochs,
                          "min_epochs": args.min_epochs, "seed": args.seed},
        )


if __name__ == "__main__":
    main()
