import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from anndata import AnnData
from kneed import KneeLocator
from scipy.sparse import isspmatrix, csr_matrix, spmatrix
from sklearn.decomposition import PCA
from typing import Optional, Tuple, List, Union

# Convert sparse matrix to dense matrix.
to_dense_matrix = lambda X: np.array(X.todense()) if isspmatrix(X) else X


def compute_pca_components(
    matrix: Union[np.ndarray, spmatrix], save_curve_img: Optional[str] = None
) -> Tuple[int, float]:
    """
    Calculate the inflection point of the PCA curve to
    obtain the number of principal components that the PCA should retain.

    Args:
        matrix: A dense or sparse matrix.
        save_curve_img: If save_curve_img != None, save the image of the PCA curve and inflection points.
    Returns:
        new_n_components: The number of principal components that PCA should retain.
        new_components_stored: Percentage of variance explained by the retained principal components.
    """
    # Convert sparse matrix to dense matrix.
    matrix = to_dense_matrix(matrix)

    # Principal component analysis (PCA).
    pca = PCA(n_components=None)
    pca.fit_transform(matrix)

    # Percentage of variance explained by each of the selected components.
    # If n_components is not set then all components are stored and the sum of the ratios is equal to 1.0.
    raw_components_ratio = pca.explained_variance_ratio_
    raw_n_components = np.arange(1, raw_components_ratio.shape[0] + 1)

    # Calculate the inflection point of the PCA curve.
    kl = KneeLocator(
        raw_n_components, raw_components_ratio, curve="convex", direction="decreasing"
    )
    new_n_components = int(kl.knee)
    new_components_stored = round(
        float(np.sum(raw_components_ratio[:new_n_components])), 3
    )

    # Whether to save the image of PCA curve and inflection point.
    if save_curve_img is not None:
        kl.plot_knee()
        plt.tight_layout()
        plt.savefig(save_curve_img, dpi=100)

    return new_n_components, new_components_stored


def qc_scanpy(
    adata: AnnData,
    min_genes: int = 200,
    min_cells: int = 3,
    mt_label: Optional[str] = "mt-",
    mt_counts_threshold: Optional[int] = None,
    total_counts_threshold: Optional[int] = None,
    save_qc_img: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Quality control based on scanpy.
    The violin image will plot the following computed quality measures:
        n_genes_by_counts: the number of genes expressed in the count matrix
        total_counts: the total counts per cell
        pct_counts_mt: the percentage of counts in mitochondrial genes

    Args:
        adata: An Anndata object.
        min_genes: Minimum number of genes expressed required for a cell to pass filtering.
        min_cells: Minimum number of cells expressed required for a gene to pass filtering.
        mt_label: Label of mitochondrial genes. For example, in Drosophila, genes whose names start with 'mt-' are mitochondrial genes.
        mt_counts_threshold: Remove cells with mitochondrial gene expression counts greater than `mt_counts_threshold`.
        total_counts_threshold: Remove cells with total counts greater than `total_counts_threshold`.
        save_qc_img: If save_qc_img != None, save the image of some computed quality measures.
        copy: Whether to copy `adata` or modify it inplace.

    Returns:
         Updates adata, containing some computed quality measures.
    """

    adata = adata.copy() if copy else adata

    # Remove duplicate genes.
    adata.var_names_make_unique()

    # Remove cells that have too many mitochondrial genes expressed or too many total counts:
    adata.var["mt"] = adata.var_names.str.startswith(mt_label)
    sc.pp.calculate_qc_metrics(
        adata, percent_top=None, qc_vars=["mt"], log1p=False, inplace=True
    )
    if mt_counts_threshold is not None:
        adata = adata[adata.obs.pct_counts_mt < mt_counts_threshold, :]
    if total_counts_threshold is not None:
        adata = adata[adata.obs.n_genes_by_counts < total_counts_threshold, :]

    # Filter genes based on number of cells or counts.
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Filter cell outliers based on counts and numbers of genes expressed.
    sc.pp.filter_cells(adata, min_genes=min_genes)

    if save_qc_img is not None:
        sc.pl.violin(
            adata,
            ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            rotation=90,
            jitter=0.4,
            multi_panel=True,
            save=False,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(save_qc_img, dpi=100)

    return adata if copy else None


def sctransform(
    adata: AnnData,
    n_top_genes: int = 3000,
    save_sct_img_1: Optional[str] = None,
    save_sct_img_2: Optional[str] = None,
    copy: bool = False,
) -> Optional[AnnData]:
    """
    Use sctransform with an additional flag vst.flavor="v2" to perform normalization and dimensionality reduction
    Original Code Repository: https://github.com/saketkc/pySCTransform

    Installation:
    Conda:
        ```conda install R```
    R:
        ```if (!require("BiocManager", quietly = TRUE))
            install.packages("BiocManager")```
        ```BiocManager::install(version = "3.14")```
        ```BiocManager::install("glmGamPoi")```
    Python:
        ```pip install rpy2```
        ```pip install git+https://github.com/saketkc/pysctransform```

    Examples:
    1.In pycharm:
        adata = sc.read_h5ad("./E16_18_S01.h5ad")
        sctransform(adata=adata, save_sct_img_1="sct10.png", save_sct_img_2="sct20.png")
        cluster_spagcn(adata=adata)
        adata.write_h5ad('test_clu.h5ad',compression='gzip')
    2.In remote server:
        os.environ['R_HOME'] = '/hwfssz1/ST_SUPERCELLS/P21Z10200N0090/tuzhencheng/software/anaconda3/envs/SCT-SpaGCN/lib/R' # Specify the R environment
        qc_scanpy(adata=adata, mt_counts_threshold=5, save_qc_img="qc0.png")
        sctransform(adata=adata, save_sct_img_1="sct10.png", save_sct_img_2="sct20.png")
        cluster_spagcn(adata=adata)
        adata.write_h5ad('test_clu.h5ad',compression='gzip')

    Args:
        adata: An Anndata object.
        n_top_genes: Number of highly-variable genes to keep.
        save_sct_img_1: If save_sct_img_1 != None, save the image of the GLM model parameters.
        save_sct_img_2: If save_sct_img_2 != None, save the image of the final residual variances.
        copy: Whether to copy `adata` or modify it inplace.

    Returns:
         Updates adata with the field ``adata.layers["raw_X"]``, containing raw expression matrix for n_top_genes(highly variable genes).
    """
    try:
        from pysctransform import vst, SCTransform
        from pysctransform.plotting import plot_fit, plot_residual_var
    except ImportError:
        raise ImportError("\nPlease install pysctransform:"
                          "\n\n\tpip install git+https://github.com/saketkc/pysctransform")

    adata = adata.copy() if copy else adata

    # Get pearson residuals for n_top_genes(highly variable genes).
    residuals = SCTransform(adata, vst_flavor="v2", var_features_n=n_top_genes)

    # Plot model characteristics.
    if save_sct_img_1 is not None or save_sct_img_2 is not None:
        # adata.obsm["pearson_residuals"] = residuals
        vst_out = vst(
            adata.X.T,
            gene_names=adata.var_names.tolist(),
            cell_names=adata.obs_names.tolist(),
            method="fix-slope",
            exclude_poisson=True,
        )
        # Visualize the GLM model parameters.
        if save_sct_img_1 is not None:
            _ = plot_fit(vst_out)
            plt.savefig(save_sct_img_1, dpi=100)
        # Visualize the final residual variances with respect to mean and highlight highly variable genes.
        if save_sct_img_2 is not None:
            _ = plot_residual_var(vst_out)
            plt.savefig(save_sct_img_2, dpi=100)

    # Only store highly variable genes.
    adata = adata[:, adata.var_names.isin(residuals.columns.tolist())]
    adata.layers["raw_X"] = adata.X

    # Highly variable genes' expression matrix.
    residuals = residuals.reindex(columns=adata.var_names.tolist())
    hvgs_matrix = (
        csr_matrix(residuals.values) if isspmatrix(adata.X) else residuals.values
    )
    adata.X = hvgs_matrix

    return adata if copy else None


def integrate(
    adatas: List[AnnData],
    batch_key: str = "slice",
) -> AnnData:
    """

    Args:
        adatas: AnnData matrices to concatenate with.
        batch_key: Add the batch annotation to :attr:`obs` using this key.

    Returns:
        integrated_adata: The concatenated AnnData, where adata.obs[batch_key] stores a categorical variable labeling the batch.
    """
    batch_ca = [adata.obs[batch_key][0] for adata in adatas]
    integrated_adata = adatas[0].concatenate(
        adatas[1:], batch_key=batch_key, batch_categories=batch_ca, join="outer"
    )
    return integrated_adata


def harmony_debatch(
    adata: AnnData,
    key: str,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    max_iter_harmony: int = 10,
    copy: bool = False,
) -> Optional[AnnData]:
    """\
    Use harmonypy [Korunsky19]_ to remove batch effects.
    This function should be run after performing PCA but before computing the neighbor graph.
    Original Code Repository: https://github.com/slowkow/harmonypy
    Interesting example: https://slowkow.com/notes/harmony-animation/

    Args:
        adata: An Anndata object.
        key: The name of the column in ``adata.obs`` that differentiates among experiments/batches.
        basis: The name of the field in ``adata.obsm`` where the PCA table is stored.
        adjusted_basis: The name of the field in ``adata.obsm`` where the adjusted PCAbtable will be stored after running this function.
        max_iter_harmony: Maximum number of rounds to run Harmony. One round of Harmony involves one clustering and one correction step.
        copy: Whether to copy `adata` or modify it inplace.

    Returns:
        Updates adata with the field ``adata.obsm[adjusted_basis]``, containing principal components adjusted by Harmony.
    """
    try:
        import harmonypy
    except ImportError:
        raise ImportError("\nPlease install harmonypy:\n\n\tpip install harmonypy")

    adata = adata.copy() if copy else adata

    # Convert sparse matrix to dense matrix.
    matrix = to_dense_matrix(adata.obsm[basis])

    # Use Harmony to adjust the PCs.
    harmony_out = harmonypy.run_harmony(
        matrix, adata.obs, key, max_iter_harmony=max_iter_harmony
    )
    adjusted_matrix = harmony_out.Z_corr.T

    # Convert dense matrix to sparse matrix.
    if isspmatrix(adata.obsm[basis]):
        adjusted_matrix = csr_matrix(adjusted_matrix)

    adata.obsm[adjusted_basis] = adjusted_matrix

    return adata if copy else None
