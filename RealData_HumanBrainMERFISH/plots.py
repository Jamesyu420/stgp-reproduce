"""Plotting style and figure helpers for Human Brain MERFISH analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Shared publication style

@dataclass(frozen=True)
class VarPartColors:
    age: str = "#E64B35"
    region: str = "#4DBBD5"
    both: str = "#3C5488"
    residuals: str = "#BFBFBF"


# Method palette (mirrors the mouse pipeline; PCA/NMF kept for back-compat).
METHOD_COLORS = {
    "stGP": "#E64B35",
    "PCA": "#91D1C2",
    "NMF": "#F39B7F",
    "SpatialPCA": "#4DBBD5",
    "MEFISTO": "#8491B4",
    "STAMP": "#B09C85",
    "Popari": "#00A087",
}


def set_nature_style(*, font: str | None = None) -> None:
    """Apply the shared publication style for all figures."""
    font_stack = [f for f in [font, "Arial", "Helvetica", "DejaVu Sans"] if f]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_stack,
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.dpi": 400,
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "lines.linewidth": 1.5,
        }
    )


# Shared helpers

DEFAULT_AGE_UNIT = "years"


def _save(fig: plt.Figure, out: str | Path | None, *, dpi: int = 400) -> None:
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")


def _bg_per_sample(adata_full, sample_ids_target) -> dict:
    """Per-sample spatial coordinates of every cell in ``adata_full``."""
    if adata_full is None:
        return {}
    bg_sp_all = np.asarray(adata_full.obsm["spatial"])
    if "id_region" not in adata_full.obs.columns:
        return {}
    bg_sample_ids = adata_full.obs["id_region"].astype(str).to_numpy()
    out: dict = {}
    for sid in np.unique(sample_ids_target):
        mask = bg_sample_ids == sid
        if mask.any():
            out[sid] = bg_sp_all[mask]
    return out


# Spatial program maps (stGP)

def _plot_spatial_programs_impl(
    *,
    adata,
    scores: pd.DataFrame,
    adata_full=None,
    age_unit: str = DEFAULT_AGE_UNIT,
    ncols: int = 5,
    bg_dot_size: float = 0.3,
    fg_dot_size: float = 4.0,
    cmap: str = "RdBu_r",
) -> list[plt.Figure]:
    """Shared backbone for stGP spatial-program tile plots."""
    obs = adata.obs
    sp = np.asarray(adata.obsm["spatial"])
    sample_ids = obs["id_region"].astype(str).to_numpy()

    if "X_stgp_spatial" in adata.obsm:
        b = np.asarray(adata.obsm["X_stgp_spatial"])
        spatial_scores = pd.DataFrame(b, index=scores.index, columns=scores.columns)
    else:
        spatial_scores = scores

    uniq_samples = np.unique(sample_ids)
    age_per_sample = np.array([
        float(obs.loc[obs["id_region"].astype(str) == s, "age"].iloc[0])
        for s in uniq_samples
    ])
    order = np.argsort(age_per_sample)
    uniq_samples = uniq_samples[order]
    age_per_sample = age_per_sample[order]
    n_samples = len(uniq_samples)

    bg_by_sample = _bg_per_sample(adata_full, sample_ids)
    sample_mask_cache: dict = {sid: sample_ids == sid for sid in uniq_samples}

    figs: list[plt.Figure] = []
    age_suffix = "yr" if age_unit == "years" else "mo"
    for prog in scores.columns.tolist():
        prog_vals = spatial_scores[prog].to_numpy(dtype=float)
        abs99 = float(np.nanpercentile(np.abs(prog_vals), 99))
        vmin, vmax = -abs99, abs99

        nrows = int(np.ceil(n_samples / ncols))
        panel_w, panel_h = 2.4, 2.4
        fig_w = ncols * panel_w + 0.8
        fig_h = nrows * panel_h + 0.5

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(fig_w, fig_h),
            gridspec_kw={"wspace": 0.04, "hspace": 0.18},
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.02,
            right=0.90,
            top=0.95,
            bottom=0.05,
            wspace=0.04,
            hspace=0.18,
        )
        axes_flat = np.atleast_1d(axes).flatten()
        for ax in axes_flat[n_samples:]:
            ax.axis("off")

        sc_ref = None
        for i, (sid, age) in enumerate(zip(uniq_samples, age_per_sample)):
            ax = axes_flat[i]
            if sid in bg_by_sample:
                bx = bg_by_sample[sid]
                ax.scatter(
                    bx[:, 0],
                    bx[:, 1],
                    c="#D8D8D8",
                    s=bg_dot_size,
                    linewidths=0,
                    rasterized=True,
                    zorder=1,
                )
            fg_mask = sample_mask_cache[sid]
            sc_ref = ax.scatter(
                sp[fg_mask, 0],
                sp[fg_mask, 1],
                c=prog_vals[fg_mask],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=fg_dot_size,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
            ax.set_aspect("equal")
            ax.set_title(f"{age:.1f} {age_suffix}", fontsize=12, pad=2)
            ax.axis("off")

        if sc_ref is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
            cbar = fig.colorbar(sc_ref, cax=cbar_ax)
            cbar.set_label(f"{prog} score")
            cbar.ax.tick_params(labelsize=9)

        figs.append(fig)

    return figs


def plot_stgp_spatial_programs(
    *,
    stgp_adata,
    scores: pd.DataFrame,
    adata_full=None,
    age_unit: str = DEFAULT_AGE_UNIT,
    ncols: int = 4,
    bg_dot_size: float = 0.3,
    fg_dot_size: float = 4.0,
    cmap: str = "RdBu_r",
) -> list[plt.Figure]:
    """One figure per stGP program, tiling all tissue sections by age."""
    return _plot_spatial_programs_impl(
        adata=stgp_adata,
        scores=scores,
        adata_full=adata_full,
        age_unit=age_unit,
        ncols=ncols,
        bg_dot_size=bg_dot_size,
        fg_dot_size=fg_dot_size,
        cmap=cmap,
    )


# Spatial kernel correlation diagnostic

def plot_spatial_kernel_corr_combined(
    adata,
    *,
    bandwidth: float,
    slice_idx: int | str | float = 0,
    age_unit: str = DEFAULT_AGE_UNIT,
    title: str | None = None,
    out: str | Path | None = None,
    dpi: int = 400,
) -> plt.Figure:
    """Two-panel diagnostic: spatial kernel matrix heatmap + per-cell scatter.

    Selects a tissue slice via ``slice_idx`` on ``obs['id_region']``.
    """
    import seaborn as sns

    ids = adata.obs["id_region"].astype(str).to_numpy()
    slices = sorted(
        pd.unique(ids),
        key=lambda sid: float(pd.to_numeric(adata.obs.loc[ids == sid, "age"], errors="coerce").iloc[0]),
    )
    if isinstance(slice_idx, (int, np.integer)) and not isinstance(slice_idx, bool):
        target_val = slices[int(slice_idx)]
    else:
        target_val = str(slice_idx)
    mask = ids == str(target_val)
    if not mask.any():
        raise ValueError(f"No rows with obs['id_region'] == {target_val!r}")

    age_note = ""
    if "age" in adata.obs.columns:
        au = pd.to_numeric(adata.obs.loc[mask, "age"], errors="coerce").dropna().unique()
        if len(au) == 1:
            age_note = f", age={float(au[0]):.1f} {age_unit}"

    coords = np.asarray(adata.obsm["spatial"][mask], dtype=float)
    n_s = len(coords)
    coords = (coords - coords.mean(0)) / np.maximum(coords.std(0, ddof=1), 1e-12)

    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(n_s, 400, replace=False)) if n_s > 400 else np.arange(n_s)
    cs = coords[sub]
    k_matrix = np.exp(-np.sum((cs[:, None] - cs[None]) ** 2, axis=2) / bandwidth)

    ref = int(np.argmin(np.linalg.norm(coords - coords.mean(0), axis=1)))
    k_vals = np.exp(-np.sum((coords - coords[ref]) ** 2, axis=1) / bandwidth)

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1])

    ax1 = fig.add_subplot(gs[0])
    sns.heatmap(k_matrix, ax=ax1, cmap="Blues", cbar_kws={"shrink": 0.8})
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"Spatial kernel matrix  (slice {slice_idx}, n={len(sub)})", pad=8)

    ax2 = fig.add_subplot(gs[1])
    sc = ax2.scatter(
        coords[:, 0],
        coords[:, 1],
        c=k_vals,
        cmap="magma",
        s=18,
        vmin=0,
        vmax=1,
        linewidths=0,
        rasterized=True,
    )
    ax2.scatter(
        coords[ref, 0],
        coords[ref, 1],
        marker="*",
        s=320,
        c="cyan",
        edgecolors="black",
        linewidths=0.6,
        zorder=10,
        label="ref cell",
    )
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    fig.colorbar(sc, ax=ax2, fraction=0.035, pad=0.02).set_label("kernel correlation")
    ax2.legend(loc="upper right", fontsize=9, frameon=False)
    ax2.set_title(
        f"Reference-cell correlation  (slice {slice_idx}, "
        f"id_region={target_val}{age_note})",
        pad=8,
    )

    if title:
        fig.suptitle(title, y=1.04)

    _save(fig, out, dpi=dpi)
    return fig


# Paper-reproduction helpers

DPI = 400


def save_pair(
    fig: plt.Figure,
    stem: str,
    *,
    out_dir: str | Path,
    dpi: int = DPI,
    bbox_inches="tight",
    pad_inches=0.04,
    display_figure: bool = True,
):
    """Save matched PNG/PDF files and optionally display the figure inline."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    kwargs = {"bbox_inches": bbox_inches}
    if pad_inches is not None:
        kwargs["pad_inches"] = pad_inches
    fig.savefig(png, dpi=dpi, **kwargs)
    fig.savefig(pdf, **kwargs)
    if display_figure:
        try:
            from IPython.display import display

            display(fig)
        except Exception:
            pass
    plt.close(fig)
    return png, pdf


def continuous_limits(vals, *, symmetric: bool = False, fixed_vmax: float | None = None) -> tuple[float, float]:
    """Robust color limits for representative spatial panels."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if fixed_vmax is not None:
        vmax = float(fixed_vmax)
        return (-vmax, vmax) if symmetric else (0.0, vmax)
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(vals), 99)) or 1.0
        return -vmax, vmax
    vmin, vmax = np.nanpercentile(vals, [1, 99]).astype(float)
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def ordered_gene_blocks(W: pd.DataFrame, *, top_n_per_program: int = 15):
    """Select non-overlapping top-loading genes in program order."""
    rows = []
    used = set()
    for program in W.index.astype(str):
        weights = W.loc[program].astype(float)
        genes = weights[weights > 0].sort_values(ascending=False).head(top_n_per_program)
        for gene_name, weight in genes.items():
            if gene_name in used:
                continue
            used.add(gene_name)
            rows.append({"program": program, "gene": str(gene_name), "anchor_weight": float(weight)})
    order = [row["gene"] for row in rows]
    return pd.DataFrame(rows), W.loc[:, order]


def ordered_stgp_alpha(info: dict, idx: int):
    """Return one stGP alpha trajectory ordered by slice age."""
    ages = np.asarray(info["ages"], dtype=float)
    alpha = np.asarray(info["alpha"], dtype=float)
    lo = np.asarray(info.get("alpha_lower", []), dtype=float)
    hi = np.asarray(info.get("alpha_upper", []), dtype=float)
    order = np.argsort(ages)
    has_ci = lo.shape == alpha.shape and hi.shape == alpha.shape
    return ages[order], alpha[idx, order], lo[idx, order] if has_ci else None, hi[idx, order] if has_ci else None, order


def draw_alpha_ci(
    ax,
    x,
    y,
    lo=None,
    hi=None,
    *,
    color: str = "#2C7FB8",
    scatter_s: float = 72,
    ci_label: str | None = "95% posterior CI",
    mean_label: str | None = "Posterior mean",
):
    """Draw the paper-style posterior age trajectory."""
    if lo is not None and hi is not None:
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0, label=ci_label)
        ax.plot(x, lo, color=color, lw=1.4, ls="--", alpha=0.65)
        ax.plot(x, hi, color=color, lw=1.4, ls="--", alpha=0.65)
    ax.plot(x, y, color=color, lw=3.0, zorder=2)
    ax.scatter(x, y, color=color, s=scatter_s, zorder=3, label=mean_label)
    ax.axhline(0, color="#8A8A8A", lw=1.0, ls=":", zorder=1)


def spatial_program_values(
    adata,
    program,
    *,
    obsm_key: str = "X_stgp_spatial",
) -> np.ndarray:
    """Extract one stGP program vector from an AnnData obsm matrix."""
    idx = int(str(program).replace("stGP", "")) - 1
    arr = np.asarray(adata.obsm[obsm_key])
    if arr.shape[0] != adata.n_obs and arr.shape[1] == adata.n_obs:
        arr = arr.T
    if arr.shape[0] != adata.n_obs:
        raise ValueError(f"{obsm_key} shape {arr.shape} is not aligned with n_obs={adata.n_obs}")
    return arr[:, idx].astype(float)


def representative_slices_by_age(
    adata,
    *,
    age_occurrences: tuple[tuple[int, int], ...] = ((28, 2), (42, 2), (82, 2), (87, 1)),
) -> list[str]:
    """Pick the Fig3 representative sections by age occurrence when available."""
    obs = adata.obs
    ids = obs["id_region"].astype(str).to_numpy()
    out = []
    for age, occurrence in age_occurrences:
        candidates = []
        for sid in pd.unique(ids):
            mask = ids == sid
            sid_age = float(pd.to_numeric(obs.loc[mask, "age"], errors="coerce").iloc[0])
            if int(round(sid_age)) == int(age):
                candidates.append(str(sid))
        candidates = sorted(candidates)
        if len(candidates) >= occurrence:
            out.append(candidates[occurrence - 1])
        elif candidates:
            out.append(candidates[-1])
    if len(out) >= 4:
        return out[:4]
    uniq = pd.unique(ids)
    return sorted(uniq, key=lambda sid: float(obs.loc[ids == sid, "age"].iloc[0]))[:4]


def plot_w_heatmap_vertical(
    W: pd.DataFrame,
    *,
    out_dir: str | Path,
    stem: str = "W_heatmap_vertical",
    top_n_per_program: int = 15,
) -> plt.Figure:
    """Paper-style vertical heatmap of top loading genes."""
    from matplotlib.colors import LinearSegmentedColormap

    block_df, W_ord = ordered_gene_blocks(W, top_n_per_program=top_n_per_program)
    mat = W_ord.T
    vmax = 0.20
    cmap = LinearSegmentedColormap.from_list("wload_human", ["#FFFFFF", "#F6B5B8", "#B2182B"])
    fig_h = max(15.2, 0.36 * len(mat.index) + 2.4)
    fig, ax = plt.subplots(figsize=(5.25, fig_h), constrained_layout=False)
    fig.subplots_adjust(left=0.31, right=0.82, top=0.985, bottom=0.070)
    im = ax.imshow(mat.to_numpy(float), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.astype(str), rotation=90, fontsize=15)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.astype(str), fontsize=12.5, fontstyle="italic")
    ax.set_xlabel("Program", fontsize=18, labelpad=8)
    ax.set_ylabel("Top loading genes", fontsize=18, labelpad=14)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    counts = block_df.groupby("program", sort=False).size().reindex(mat.columns, fill_value=0)
    y = 0
    for count in counts:
        y += int(count)
        if y < mat.shape[0]:
            ax.axhline(y - 0.5, color="white", lw=1.2)

    cax = fig.add_axes([0.855, 0.20, 0.040, 0.60])
    cbar = fig.colorbar(im, cax=cax, ticks=[0, 0.05, 0.10, 0.15, 0.20])
    cbar.ax.set_yticklabels(["0", "0.05", "0.10", "0.15", "0.20"])
    cbar.set_label("Gene weight W", fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=13, length=4)
    save_pair(fig, stem, out_dir=out_dir, pad_inches=0.03)
    return fig


def plot_alpha_trajectory(
    stgp_info: dict,
    program_idx: int,
    *,
    out_dir: str | Path,
    stem: str | None = None,
    age_label: str = "Age (yr)",
) -> plt.Figure:
    """Paper-style single-program alpha trajectory."""
    ages_ord, y, lo, hi, _ = ordered_stgp_alpha(stgp_info, program_idx)
    fig, ax = plt.subplots(figsize=(6.5, 5.05), constrained_layout=True)
    draw_alpha_ci(ax, ages_ord, y, lo, hi, color="#2C7FB8", scatter_s=72)
    ax.set_xlabel(age_label, fontsize=24)
    ax.set_ylabel(r"Aging effect $\alpha$", fontsize=24)
    ax.tick_params(axis="both", labelsize=20, length=4.5, width=1.1)
    if lo is not None and hi is not None:
        ax.legend(fontsize=18, loc="best")
    if stem is None:
        stem = f"alpha_trajectory_stGP{program_idx + 1}"
    save_pair(fig, stem, out_dir=out_dir)
    return fig


def plot_representative_continuous_tiles(
    adata,
    values,
    *,
    rep_slices: list[str] | None = None,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str = "score",
    title: str | None = None,
    out_dir: str | Path,
    stem: str,
    dot_size: float = 2.3,
) -> plt.Figure:
    """2 x 2 representative-slice scatter tiles for Fig3/Fig4-style panels."""
    values = np.asarray(values, dtype=float)
    if rep_slices is None:
        rep_slices = representative_slices_by_age(adata)
    ids = adata.obs["id_region"].astype(str).to_numpy()
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    if vmin is None or vmax is None:
        lim = continuous_limits(values, symmetric=symmetric)
        vmin = lim[0] if vmin is None else vmin
        vmax = lim[1] if vmax is None else vmax

    fig, axes = plt.subplots(2, 2, figsize=(3.70, 3.58), squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.84, top=0.90, bottom=0.02, wspace=0.02, hspace=0.12)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sid in zip(axes.ravel(), rep_slices):
        mask = ids == str(sid)
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=values[mask],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=dot_size,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
    cbar_ax = fig.add_axes([0.875, 0.18, 0.030, 0.64])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label(colorbar_label, fontsize=10, labelpad=4)
    cb.ax.tick_params(labelsize=9, length=2.5, width=0.8)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)
    save_pair(fig, stem, out_dir=out_dir, pad_inches=0.01)
    return fig


def plot_spatial_program_all_slices(
    adata,
    program,
    *,
    out_dir: str | Path,
    stem: str | None = None,
    ncols: int = 4,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    dot_size: float = 2.2,
    title: str | None = None,
) -> plt.Figure:
    """Fig4-style all-slice spatial grid for one stGP spatial field."""
    values = spatial_program_values(adata, program)
    ids = adata.obs["id_region"].astype(str).to_numpy()
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    slices = sorted(
        pd.unique(ids),
        key=lambda sid: float(pd.to_numeric(adata.obs.loc[ids == sid, "age"], errors="coerce").iloc[0]),
    )
    ages = {
        sid: float(pd.to_numeric(adata.obs.loc[ids == sid, "age"], errors="coerce").iloc[0])
        for sid in slices
    }
    vmin, vmax = continuous_limits(values, symmetric=symmetric)
    nrows = int(np.ceil(len(slices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.15 * ncols + 0.55, 2.0 * nrows), squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.92, bottom=0.03, wspace=0.02, hspace=0.13)
    for ax in axes.ravel():
        ax.axis("off")
    sc_ref = None
    for ax, sid in zip(axes.ravel(), slices):
        mask = ids == str(sid)
        sc_ref = ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=values[mask],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=dot_size,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_title(f"{ages[sid]:.0f} yr", fontsize=10, pad=2)
    if sc_ref is not None:
        cax = fig.add_axes([0.905, 0.18, 0.018, 0.64])
        cb = fig.colorbar(sc_ref, cax=cax)
        cb.set_label("score", fontsize=10)
        cb.ax.tick_params(labelsize=9, length=2.5, width=0.8)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)
    if stem is None:
        stem = f"spatial_{program}"
    save_pair(fig, stem, out_dir=out_dir, pad_inches=0.015)
    return fig


# HumanBrain ext Fig. 3 reproduction panels

def p_to_stars(p: float, *, nan_label: str = "NA", nonsig_label: str = "ns") -> str:
    """Convert a p-value to compact significance stars."""
    if not np.isfinite(p):
        return nan_label
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return nonsig_label


def _clean_layer_labels(values, *, default: str = "ext") -> np.ndarray:
    vals = pd.Series(values).astype("string").fillna(default).astype(str)
    return vals.replace({"#unassigned": default, "unassigned": default, "<NA>": default}).to_numpy()


def _slice_xy(adata):
    ids = adata.obs["id_region"].astype(str).to_numpy()
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    return ids, xy


def _representative_axis(ax, xy, mask, *, dot_size: float):
    ax.set_aspect("equal")
    ax.axis("off")
    return xy[mask, 0], xy[mask, 1]


def plot_layer_cluster_tiles(
    adata,
    values,
    *,
    rep_slices: Sequence[str],
    palette: Mapping[str, str],
    out_dir: str | Path,
    stem: str,
    title: str | None = None,
    legend_items: Sequence[tuple[str, str]] | None = None,
    show_legend: bool = False,
    dot_size: float = 2.3,
) -> plt.Figure:
    """Plot 2 x 2 representative layer-label spatial tiles."""
    ids, xy = _slice_xy(adata)
    labels = _clean_layer_labels(values)
    fig, axes = plt.subplots(2, 2, figsize=(3.55, 3.58 if title else 3.42), squeeze=False)
    fig.subplots_adjust(
        left=0.02,
        right=0.76 if show_legend else 0.98,
        top=0.90 if title else 0.96,
        bottom=0.02,
        wspace=0.02,
        hspace=0.12,
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, sid in zip(axes.ravel(), rep_slices):
        mask = ids == str(sid)
        x, y = _representative_axis(ax, xy, mask, dot_size=dot_size)
        ax.scatter(
            x,
            y,
            c=[palette.get(label, "#888888") for label in labels[mask]],
            s=dot_size,
            linewidths=0,
            rasterized=True,
        )
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)
    if show_legend and legend_items:
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=palette[key], edgecolor="none", label=label)
            for key, label in legend_items
        ]
        fig.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(0.78, 0.50),
            frameon=False,
            fontsize=11.5,
            handlelength=1.0,
            handletextpad=0.45,
        )
    save_pair(fig, stem, out_dir=out_dir, pad_inches=0.01)
    return fig


def plot_representative_embedding_tiles(
    adata,
    values,
    *,
    rep_slices: Sequence[str],
    out_dir: str | Path,
    stem: str,
    title: str | None = None,
    signed: bool | None = None,
    cmap: str | None = None,
    colorbar_label: str = "score",
    dot_size: float = 2.3,
) -> plt.Figure:
    """Plot a continuous embedding over the four representative human sections."""
    vals = np.asarray(values, dtype=float)
    if signed is None:
        signed = np.nanmin(vals) < 0
    if cmap is None:
        cmap = "RdBu_r" if signed else "YlOrBr"
    vmin, vmax = continuous_limits(vals, symmetric=signed)
    return plot_representative_continuous_tiles(
        adata,
        vals,
        rep_slices=list(rep_slices),
        cmap=cmap,
        symmetric=signed,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=colorbar_label,
        title=title,
        out_dir=out_dir,
        stem=stem,
        dot_size=dot_size,
    )


def _plot_cluster_cell(fig, subspec, adata, labels, rep_slices, palette, *, dot_size: float = 1.8) -> None:
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    ids, xy = _slice_xy(adata)
    labels = _clean_layer_labels(labels)
    inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=subspec, wspace=0.025, hspace=0.045)
    for i, sid in enumerate(rep_slices):
        ax = fig.add_subplot(inner[i // 2, i % 2])
        mask = ids == str(sid)
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=[palette.get(label, "#888888") for label in labels[mask]],
            s=dot_size,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.axis("off")


def _plot_embedding_cell(fig, subspec, adata, values, rep_slices, *, signed: bool, dot_size: float = 1.8) -> None:
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    ids, xy = _slice_xy(adata)
    vals = np.asarray(values, dtype=float)
    vmin, vmax = continuous_limits(vals, symmetric=signed)
    cmap = "RdBu_r" if signed else "YlOrBr"
    inner = GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=subspec,
        width_ratios=[1, 1, 0.08],
        wspace=0.045,
        hspace=0.045,
    )
    for i, sid in enumerate(rep_slices):
        ax = fig.add_subplot(inner[i // 2, i % 2])
        mask = ids == str(sid)
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=vals[mask],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=dot_size,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.axis("off")
    cax = fig.add_subplot(inner[:, 2])
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("score", fontsize=8.5, labelpad=2)
    cb.ax.tick_params(labelsize=7.5, length=2.0, width=0.7)


def plot_human_cluster_embedding_composite(
    adata,
    *,
    rep_slices: Sequence[str],
    cluster_values: Mapping[str, Sequence],
    embedding_values: Mapping[str, Sequence[float]],
    methods: Sequence[str],
    palette: Mapping[str, str],
    legend_items: Sequence[tuple[str, str]],
    signed_methods: set[str] | None = None,
    out_dir: str | Path,
    stem: str = "human_cluster_embedding_4x2_panel",
) -> plt.Figure | None:
    """Composite of baseline layer labels and L4/RORB-associated embeddings."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    signed_methods = {"MEFISTO", "SpatialPCA"} if signed_methods is None else set(signed_methods)
    available = [m for m in methods if m in cluster_values and m in embedding_values]
    if not available:
        print("[skip] Human baseline cluster/embedding composite: no complete baseline inputs were available.")
        return None

    fig = plt.figure(figsize=(7.9, 2.98 * len(available)), constrained_layout=False)
    outer = GridSpec(
        len(available),
        2,
        figure=fig,
        left=0.055,
        right=0.965,
        top=0.965,
        bottom=0.085,
        width_ratios=[1.0, 1.10],
        hspace=0.26,
        wspace=0.11,
    )
    for row_i, method in enumerate(available):
        _plot_cluster_cell(fig, outer[row_i, 0], adata, cluster_values[method], rep_slices, palette)
        vals = np.asarray(embedding_values[method], dtype=float)
        signed = method in signed_methods or np.nanmin(vals) < 0
        _plot_embedding_cell(fig, outer[row_i, 1], adata, vals, rep_slices, signed=signed)
        left = outer[row_i, 0].get_position(fig)
        right = outer[row_i, 1].get_position(fig)
        fig.text(
            (left.x0 + right.x1) / 2,
            max(left.y1, right.y1) + 0.014,
            method,
            ha="center",
            va="bottom",
            fontsize=13.5,
            fontweight="bold",
        )

    handles = [Patch(facecolor=palette[key], edgecolor="none", label=label) for key, label in legend_items]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.020),
        fontsize=12.5,
        handlelength=1.0,
        handletextpad=0.45,
        columnspacing=1.3,
    )
    save_pair(fig, stem, out_dir=out_dir, pad_inches=0.015)
    return fig


def _metric_boxplot(
    ax,
    df: pd.DataFrame,
    metric: str,
    y_label: str,
    *,
    methods: Sequence[str],
    fontsize: int = 11,
    add_brackets: bool = True,
) -> None:
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    data = [df.loc[df["method"] == method, metric].dropna().to_numpy(float) for method in methods]
    colors = [METHOD_COLORS.get(method, "#999999") for method in methods]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", linewidth=1.3),
        whiskerprops=dict(linewidth=0.9),
        capprops=dict(linewidth=0.9),
        flierprops=dict(marker="o", markersize=3.0, alpha=0.45, markeredgewidth=0),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=fontsize)
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize + 2)
    ax.tick_params(axis="y", labelsize=fontsize, length=3.2, width=0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not add_brackets or "stGP" not in methods:
        return

    ref = df.loc[df["method"] == "stGP", ["id_region", metric]].rename(columns={metric: "ref"})
    comps, raw_p = [], []
    for method in methods:
        if method == "stGP":
            continue
        other = df.loc[df["method"] == method, ["id_region", metric]].rename(columns={metric: "other"})
        paired = ref.merge(other, on="id_region").dropna()
        diff = paired["ref"].to_numpy(float) - paired["other"].to_numpy(float)
        if len(diff) < 2:
            p = np.nan
        elif np.allclose(diff, 0):
            p = 1.0
        else:
            p = float(wilcoxon(paired["ref"], paired["other"], alternative="greater").pvalue)
        comps.append(method)
        raw_p.append(p)

    raw_p = np.asarray(raw_p, dtype=float)
    adj = np.full_like(raw_p, np.nan)
    valid = np.isfinite(raw_p)
    if valid.any():
        _, adj[valid], _, _ = multipletests(raw_p[valid], method="holm")
    vals = np.concatenate([d[np.isfinite(d)] for d in data if np.isfinite(d).any()])
    if vals.size == 0:
        return
    ymin, ymax = float(np.nanmin(vals)), float(np.nanmax(vals))
    yr = ymax - ymin if ymax > ymin else 1.0
    y0 = ymax + 0.08 * yr
    step = 0.12 * yr
    dy = 0.03 * yr
    for i, (method, p_adj) in enumerate(zip(comps, adj)):
        x1, x2 = 1, list(methods).index(method) + 1
        y = y0 + i * step
        ax.plot([x1, x1, x2, x2], [y, y + dy, y + dy, y], lw=0.85, color="black", clip_on=False)
        ax.text((x1 + x2) / 2, y + dy, p_to_stars(p_adj), ha="center", va="bottom", fontsize=fontsize - 1)
    ax.set_ylim(top=y0 + len(comps) * step + 0.10 * yr)


def plot_marker_region_ari_nmi_combined(
    csv_path: str | Path,
    *,
    methods: Sequence[str],
    out_dir: str | Path,
    stem: str = "human_marker_region_ARI_NMI_1x2",
) -> plt.Figure:
    """Draw the Fig. 3 marker-region ARI/NMI benchmark panel."""
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.85), constrained_layout=True)
    _metric_boxplot(axes[0], df, "raw_ari", "", methods=methods, fontsize=12)
    _metric_boxplot(axes[1], df, "raw_nmi", "", methods=methods, fontsize=12)
    axes[0].set_title("ARI", fontsize=18, pad=8)
    axes[1].set_title("NMI", fontsize=18, pad=8)
    save_pair(fig, stem, out_dir=out_dir)
    return fig


def _add_layer_marker_title(ax, layer_name: str, marker_gene: str) -> None:
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

    normal_left = TextArea(f"{layer_name} (", textprops={"fontsize": 15, "fontfamily": "Arial"})
    gene_text = TextArea(marker_gene, textprops={"fontsize": 15, "fontfamily": "Arial", "fontstyle": "italic"})
    normal_right = TextArea(")", textprops={"fontsize": 15, "fontfamily": "Arial"})
    title_box = HPacker(children=[normal_left, gene_text, normal_right], align="center", pad=0, sep=0)
    anchored = AnchoredOffsetbox(
        loc="upper center",
        child=title_box,
        frameon=False,
        pad=0,
        borderpad=0,
        bbox_to_anchor=(0.5, 1.12),
        bbox_transform=ax.transAxes,
    )
    ax.add_artist(anchored)


def plot_embedding_marker_correlation_boxplots(
    csv_path: str | Path,
    *,
    methods: Sequence[str],
    layer_markers: Mapping[str, str],
    out_dir: str | Path,
    stem: str = "human_embedding_vs_marker_correlation_all_layers",
) -> plt.Figure:
    """Draw the Fig. 3 layer-marker correlation benchmark panel."""
    corr = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.95), constrained_layout=True)
    for ax, layer_name in zip(axes, ["L2/3", "L4", "L5/6"]):
        sub = corr[corr["layer"] == layer_name].copy().rename(columns={"correlation": "corr"})
        _metric_boxplot(
            ax,
            sub,
            "corr",
            "Pearson correlation" if ax is axes[0] else "",
            methods=methods,
            fontsize=11,
        )
        _add_layer_marker_title(ax, layer_name, layer_markers[layer_name])
    save_pair(fig, stem, out_dir=out_dir)
    return fig
