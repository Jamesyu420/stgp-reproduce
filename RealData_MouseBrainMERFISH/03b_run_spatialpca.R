#!/usr/bin/env Rscript
# ----------------------------------------------------------------------------
# Step 03b: Multi-sample SpatialPCA on MERFISH AnnData (R-only baseline).
#
# Mirrors SpatialPCA_Multiple_Sample (Shang & Zhou, v1.2.0). Deviations:
#   * Skip SPARK SVG selection (irrelevant for a 220-gene MERFISH panel) ->
#     use the gene intersection across samples instead.
#   * Sparse block-diagonal kernel (dense N x N OOMs at >15k cells).
#   * Positional per-sample slicing of joint SpatialPCs (the original
#     grep("1_<i>", ...) misassigns cells when N >= 10 samples).
#   * Seurat rPCA reduction (recommended for >=5 datasets).
#
# Usage:
#   Rscript 03b_run_spatialpca.R --all
#   Rscript 03b_run_spatialpca.R --celltypes Microglia OPC
#   Rscript 03b_run_spatialpca.R --input X.h5ad --out-dir Y/
# ----------------------------------------------------------------------------

suppressMessages({
    library(anndataR); library(Seurat); library(SpatialPCA)
    library(Matrix);   library(jsonlite)
})

BASELINE_RUN_ORDER <- c(
    "T cell", "NSC", "Neuroblast", "Macrophage", "Ependymal", "VSMC", "Pericyte",
    "OPC", "Microglia", "Endothelial", "Astrocyte", "Neuron-MSN",
    "Oligodendrocyte", "Neuron-Excitatory"
)

DATA_DIR_DEFAULT   <- "data/processed"
OUT_ROOT_DEFAULT   <- "Results/baselines/spatialpca"
TIMING_LOG_DEFAULT <- "Results/benchmark_runtimes.jsonl"

# Hyperparameters that are no longer CLI-exposed (rarely touched in practice).
EIGENVECNUM_REQUESTED <- 100L     # clamped to n_total - 1 inside run_one
SPARSE_KERNEL_TOL     <- 1e-5
SPARSE_KERNEL_NCORE   <- 8L
NFEATURES_ANCHOR      <- 2000L
INTEGRATION_DIMS      <- 1:30
INTEGRATION_K_ANCHOR  <- 5L
INTEGRATION_K_FILTER  <- 200L


# ════════════════════════════════════════════════════════════════════════════
# Small helpers
# ════════════════════════════════════════════════════════════════════════════

safe_name <- function(ct) gsub("[^A-Za-z0-9_-]", "_", ct)

is_completed <- function(p) {
    if (!file.exists(p)) return(FALSE)
    j <- tryCatch(jsonlite::fromJSON(p), error = function(e) NULL)
    isTRUE(identical(j$status, "completed"))
}

append_jsonl <- function(p, rec) {
    dir.create(dirname(p), recursive = TRUE, showWarnings = FALSE)
    cat(jsonlite::toJSON(rec, auto_unbox = TRUE, na = "null"), "\n",
        sep = "", file = p, append = TRUE)
}


# ════════════════════════════════════════════════════════════════════════════
# Multi-sample SpatialPCA (SPARK skipped, sparse kernel, positional slicing).
# ════════════════════════════════════════════════════════════════════════════

spatialpca_multi <- function(count_list, location_list,
                              k = 4, bandwidth_common = 0.1, eigenvecnum = NULL) {
    n <- length(count_list); stopifnot(n == length(location_list))
    bandwidth_common <- as.numeric(bandwidth_common)

    # Force Seurat v3 assays (SpatialPCA reads @assays$integrated@scale.data).
    old_av <- getOption("Seurat.object.assay.version", "v5")
    options(Seurat.object.assay.version = "v3")
    on.exit(options(Seurat.object.assay.version = old_av), add = TRUE)

    # 1. Globally unique cell ids: Sample{i}_<orig>.
    for (i in seq_len(n)) {
        nm <- paste0("Sample", i, "_", colnames(count_list[[i]]))
        colnames(count_list[[i]])    <- nm
        rownames(location_list[[i]]) <- nm
    }

    # 2. Common gene set (skip SPARK).
    common <- Reduce(intersect, lapply(count_list, rownames))
    nf <- min(NFEATURES_ANCHOR, length(common))
    cat(sprintf("[spatialpca] %d common genes across %d samples.\n",
                length(common), n))

    # 3. Per-sample LogNormalize + variable features + per-sample sparse kernel.
    seu_list    <- vector("list", n)
    loc_list    <- vector("list", n)
    kernel_list <- vector("list", n)
    for (i in seq_len(n)) {
        s <- CreateSeuratObject(counts = count_list[[i]][common, , drop = FALSE])
        s <- NormalizeData(s, verbose = FALSE)
        s <- FindVariableFeatures(s, selection.method = "vst",
                                  nfeatures = nf, verbose = FALSE)
        seu_list[[i]] <- s
        loc_list[[i]] <- as.matrix(location_list[[i]][colnames(s), , drop = FALSE])

        kernel_list[[i]] <- kernel_build_sparse(
            kerneltype = "gaussian", location = scale(loc_list[[i]]),
            bandwidth  = bandwidth_common, tol = SPARSE_KERNEL_TOL,
            ncores     = SPARSE_KERNEL_NCORE)
        cat(sprintf("[spatialpca] sample %d/%d: %d cells, bw=%.4g\n",
                    i, n, ncol(s), bandwidth_common))
    }

    # 4. Seurat rPCA integration -> joint normalized expression matrix.
    cat("[spatialpca] Seurat rPCA integration...\n")
    feats <- Seurat::SelectIntegrationFeatures(
        object.list = seu_list, nfeatures = nf, verbose = FALSE)
    seu_list <- lapply(seu_list, function(s) {
        s <- ScaleData(s, features = feats, verbose = FALSE)
        s <- RunPCA(s, features = feats,
                    npcs = max(INTEGRATION_DIMS), verbose = FALSE)
        s
    })
    anchors <- Seurat::FindIntegrationAnchors(
        object.list      = seu_list, anchor.features = feats,
        reduction        = "rpca",   dims            = INTEGRATION_DIMS,
        k.anchor         = INTEGRATION_K_ANCHOR,
        k.filter         = INTEGRATION_K_FILTER,
        verbose          = FALSE)
    combined <- Seurat::IntegrateData(
        anchorset = anchors, dims = INTEGRATION_DIMS, verbose = FALSE)
    DefaultAssay(combined) <- "integrated"
    combined <- Seurat::ScaleData(combined, verbose = FALSE)
    integrated <- as.matrix(GetAssayData(
        combined, assay = "integrated", layer = "scale.data"))

    # 5. Joint expression in original Sample{i}_<orig> order, per-gene z-score.
    all_cells <- unlist(lapply(seu_list, colnames), use.names = FALSE)
    expr <- t(scale(t(integrated[, all_cells, drop = FALSE])))
    expr <- expr[is.finite(rowSums(expr)), , drop = FALSE]
    rm(combined, anchors, integrated); invisible(gc(verbose = FALSE))

    # 6. Assemble SpatialPCA object (block-diagonal sparse kernel).
    obj <- new("SpatialPCA",
        counts             = as(do.call(cbind, lapply(seq_len(n), function(i)
                                count_list[[i]][rownames(expr), colnames(seu_list[[i]]),
                                                drop = FALSE])), "dgCMatrix"),
        normalized_expr    = expr,
        project            = "MultipleSample",
        location           = do.call(rbind, loc_list),
        kernelmat          = Matrix::bdiag(kernel_list),
        kerneltype         = "gaussian",
        bandwidthtype      = "user",
        bandwidth          = bandwidth_common,
        sparseKernel       = TRUE,
        sparseKernel_tol   = SPARSE_KERNEL_TOL,
        sparseKernel_ncore = SPARSE_KERNEL_NCORE,
        fast               = TRUE,
        eigenvecnum        = if (is.null(eigenvecnum)) 0 else as.numeric(eigenvecnum),
        SpatialPCnum       = k, tau = 1, sigma2_0 = 1, params = list())
    obj@params$expr <- expr

    # 7. SpatialPCA core.
    cat(sprintf("[spatialpca] EstimateLoading + SpatialPCs (k=%d, eigenvecnum=%s)...\n",
                k, ifelse(is.null(eigenvecnum), "auto", as.character(eigenvecnum))))
    obj <- SpatialPCA_EstimateLoading(obj, fast = TRUE,
                                      eigenvecnum  = eigenvecnum,
                                      SpatialPCnum = k)
    obj <- SpatialPCA_SpatialPCs(obj, fast = TRUE, eigenvecnum = eigenvecnum)
    colnames(obj@SpatialPCs) <- rownames(obj@location)
    obj
}


# ════════════════════════════════════════════════════════════════════════════
# Single-celltype runner: AnnData -> per-mouse lists -> fit -> outputs.
# ════════════════════════════════════════════════════════════════════════════

run_one <- function(input_h5ad, out_dir, k, group_col, bandwidth, seed) {
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    set.seed(seed); options(future.globals.maxSize = 8 * 1024^3)

    cat(sprintf("[spatialpca] reading %s\n", input_h5ad))
    adata <- read_h5ad(input_h5ad)
    obs_names <- adata$obs_names
    var_names <- make.unique(adata$var_names)
    if (!(group_col %in% colnames(adata$obs)))
        stop(sprintf("obs[%s] not found.", group_col))
    if (!("spatial" %in% names(adata$obsm)))
        stop("obsm[`spatial`] not found.")

    expr <- as(adata$X, "dgCMatrix")
    rownames(expr) <- obs_names; colnames(expr) <- var_names
    count_all <- as(t(expr), "dgCMatrix")             # genes x cells
    coord_all <- as.matrix(adata$obsm[["spatial"]])
    rownames(coord_all) <- obs_names; colnames(coord_all) <- c("x", "y")
    sample_id <- as.character(adata$obs[[group_col]])

    # Per-sample lists (drop samples with <2 cells).
    count_list <- list(); location_list <- list(); sample_used <- character()
    for (s in sort(unique(sample_id))) {
        ids <- obs_names[sample_id == s]
        if (length(ids) < 2L) {
            cat(sprintf("[spatialpca] skipping sample %s (only %d cells).\n",
                        s, length(ids))); next
        }
        count_list[[length(count_list) + 1]]       <- count_all[, ids, drop = FALSE]
        location_list[[length(location_list) + 1]] <- coord_all[ids, , drop = FALSE]
        sample_used <- c(sample_used, s)
    }
    if (length(count_list) < 2)
        stop("Multi-sample SpatialPCA needs >= 2 samples.")

    # eigs_sym requires k < n; clamp eigenvecnum to n_total - 1.
    n_total <- sum(sapply(count_list, ncol))
    eig <- max(2L, min(EIGENVECNUM_REQUESTED, as.integer(n_total - 1L)))
    cat(sprintf("[spatialpca] %d cells x %d genes x %d samples; k=%d, bandwidth=%.4g\n",
                nrow(adata$X), ncol(adata$X), length(count_list), k, bandwidth))

    obj <- spatialpca_multi(count_list, location_list,
                             k = k, bandwidth_common = bandwidth, eigenvecnum = eig)
    sp  <- obj@SpatialPCs                              # k x n_kept
    pc_cols <- paste0("SPCA", seq_len(nrow(sp)))

    # Map joint cells back to original AnnData order.
    orig_cells <- sub("^Sample[0-9]+_", "", colnames(sp))
    if (any(duplicated(orig_cells)))
        stop("Original cell barcodes are not unique after Sample-prefix stripping.")
    Z <- matrix(0, nrow = nrow(coord_all), ncol = nrow(sp),
                dimnames = list(obs_names, pc_cols))
    idx <- match(orig_cells, obs_names)
    if (any(is.na(idx)))
        stop(sprintf("%d joint cells not in obs_names.", sum(is.na(idx))))
    Z[idx, ] <- t(sp)

    # ----- outputs (3 files: adata, W, run_config) -------------------------
    adata$obsm[["X_spatialpca"]] <- Z
    out_h5ad <- file.path(out_dir, "adata_with_scores.h5ad")
    if (file.exists(out_h5ad)) file.remove(out_h5ad)
    adata$write_h5ad(out_h5ad)

    W <- obj@W
    rownames(W) <- rownames(obj@normalized_expr); colnames(W) <- pc_cols
    write.csv(W, file.path(out_dir, "W_loadings.csv"), row.names = TRUE)

    cfg <- list(
        method                = "SpatialPCA",
        input                 = input_h5ad, out_dir = out_dir,
        k                     = k, group_col = group_col, seed = seed,
        bandwidth_common      = bandwidth,
        eigenvecnum           = eig,
        nfeatures_anchor      = NFEATURES_ANCHOR,
        integration_dims      = max(INTEGRATION_DIMS),
        integration_reduction = "rpca",
        sparse_kernel_tol     = SPARSE_KERNEL_TOL,
        sparse_kernel_ncore   = SPARSE_KERNEL_NCORE,
        n_samples             = length(count_list),
        n_cells               = ncol(sp),
        n_genes_used          = nrow(W),
        sample_map            = data.frame(
            group_value = sample_used,
            spca_name   = paste0("Sample", seq_along(sample_used)),
            n_cells     = sapply(count_list, ncol),
            stringsAsFactors = FALSE),
        seurat_version        = as.character(packageVersion("Seurat")),
        spatialpca_version    = as.character(packageVersion("SpatialPCA"))
    )
    write_json(cfg, file.path(out_dir, "run_config.json"),
               auto_unbox = TRUE, pretty = TRUE, dataframe = "rows")
    cat(sprintf("[spatialpca] wrote outputs in %s\n", out_dir))
}


# ════════════════════════════════════════════════════════════════════════════
# CLI / pipeline driver.
# ════════════════════════════════════════════════════════════════════════════

parse_args <- function(argv = commandArgs(trailingOnly = TRUE)) {
    args <- list(all = FALSE, celltypes = NULL, celltype = NULL,
                 input = NULL, `out-dir` = NULL,
                 `data-dir` = DATA_DIR_DEFAULT, `out-dir-root` = OUT_ROOT_DEFAULT,
                 `timing-log` = TIMING_LOG_DEFAULT,
                 k = "4", `group-col` = "mouse_id",
                 bandwidth = "0.1", seed = "1234")
    flags     <- c("all")
    nargs_one <- c("celltypes")
    i <- 1
    while (i <= length(argv)) {
        key <- sub("^--", "", argv[i])
        if (key %in% flags) { args[[key]] <- TRUE; i <- i + 1 }
        else if (key %in% nargs_one) {
            j <- i + 1; vals <- character()
            while (j <= length(argv) && !startsWith(argv[j], "--")) {
                vals <- c(vals, argv[j]); j <- j + 1
            }
            args[[key]] <- vals; i <- j
        } else { args[[key]] <- argv[i + 1]; i <- i + 2 }
    }
    args
}

run_with_timing <- function(ct, input_h5ad, out_dir,
                             k, group_col, bandwidth, seed, timing_log) {
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    timing_path <- file.path(out_dir, "timing.json")
    if (is_completed(timing_path)) {
        message(sprintf("[SKIP] spatialpca for %s already done.", ct))
        return(invisible(NULL))
    }
    t0 <- Sys.time(); status <- "completed"
    tryCatch(run_one(input_h5ad, out_dir, k, group_col, bandwidth, seed),
             error = function(e) {
                 status <<- sprintf("error (%s)", conditionMessage(e))
                 message(sprintf("[ERROR] %s: %s", ct, conditionMessage(e)))
                 cat(paste(capture.output(traceback()), collapse = "\n"), "\n",
                     sep = "")
             })
    elapsed <- as.numeric(Sys.time() - t0, units = "secs")
    timing <- list(method = "spatialpca", celltype = ct,
                   runtime_sec = round(elapsed, 2), status = status,
                   k = k, bandwidth = bandwidth)
    write_json(timing, timing_path, auto_unbox = TRUE, pretty = TRUE)
    if (!is.null(timing_log)) append_jsonl(timing_log, list(
        celltype = ct, method = "spatialpca",
        runtime_sec = round(elapsed, 2), status = status,
        out_dir = normalizePath(out_dir, mustWork = FALSE)))
    cat(sprintf("[spatialpca] %s -> %s (%.1fs)\n", ct, status, elapsed))
}

main <- function() {
    args <- parse_args()
    k    <- as.integer(args$k)
    seed <- as.integer(args$seed)
    group_col <- args[["group-col"]]
    bandwidth <- as.numeric(args$bandwidth)
    if (!is.finite(bandwidth) || bandwidth <= 0)
        stop(sprintf("--bandwidth must be a positive number; got '%s'.", args$bandwidth))
    timing_log <- args[["timing-log"]]
    if (identical(timing_log, "") || tolower(timing_log) %in% c("none", "null"))
        timing_log <- NULL

    # Resolve cell types: --celltype X > --celltypes X Y > --all > [].
    cts <- if (!is.null(args$celltype))     args$celltype
           else if (!is.null(args$celltypes)) args$celltypes
           else if (isTRUE(args$all))         BASELINE_RUN_ORDER
           else                               character(0)

    # Single-celltype mode (no --all / --celltype(s)).
    if (length(cts) == 0L) {
        if (is.null(args$input) || is.null(args[["out-dir"]]))
            stop("Single mode requires --input and --out-dir.")
        ct <- basename(tools::file_path_sans_ext(args$input))
        run_with_timing(ct, args$input, args[["out-dir"]],
                         k, group_col, bandwidth, seed, timing_log)
        return(invisible(NULL))
    }

    for (ct in cts) {
        input_h5ad <- if (length(cts) == 1L && !is.null(args$input))
                          args$input
                      else
                          file.path(args[["data-dir"]], paste0(safe_name(ct), ".h5ad"))
        out_dir    <- if (length(cts) == 1L && !is.null(args[["out-dir"]]))
                          args[["out-dir"]]
                      else
                          file.path(args[["out-dir-root"]], safe_name(ct))
        if (!file.exists(input_h5ad)) {
            message(sprintf("[ERROR] missing %s; skipping %s.", input_h5ad, ct))
            next
        }
        cat(sprintf("\n%s\n[spatialpca] %s\n%s\n",
                    strrep("=", 60), ct, strrep("=", 60)))
        run_with_timing(ct, input_h5ad, out_dir,
                         k, group_col, bandwidth, seed, timing_log)
    }
    cat("\n=== SPATIALPCA BASELINE COMPLETE ===\n")
}

main()
