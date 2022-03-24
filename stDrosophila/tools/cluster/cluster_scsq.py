import pandas as pd
import scanpy as sc
import squidpy as sq

from anndata import AnnData
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .utils import compute_pca_components, harmony_debatch


def spatial_adj_scsq(
    adata: AnnData,
    spatial_key: str = "spatial",
    pca_key: str = "X_pca",
    e_neigh: int = 30,
    s_neigh: int = 6,
):
    """Calculate the adjacent matrix based on a neighborhood graph of gene expression space and a neighborhood graph of physical space."""

    # Compute a neighborhood graph of gene expression space.
    sc.pp.neighbors(adata, n_neighbors=e_neigh, use_rep=pca_key)

    # Compute a neighborhood graph of physical space.
    sq.gr.spatial_neighbors(adata, n_neighs=s_neigh, spatial_key=spatial_key)

    # Calculate the adjacent matrix.
    conn = adata.obsp["connectivities"].copy()
    conn.data[conn.data > 0] = 1
    adj = conn + adata.obsp["spatial_connectivities"]
    adj.data[adj.data > 0] = 1
    return adj


def cluster_scsq(
    adata: AnnData,
    spatial_key: str = "spatial",
    key_added: Optional[str] = "leiden",
    n_pca_components: Optional[int] = None,
    e_neigh: int = 30,
    s_neigh: int = 6,
    cluster_method: Literal["leiden", "louvain"] = "leiden",
    resolution: float = 0.8,
    debatch: bool = False,
    batch_key: Optional[str] = "slice",
    max_iter_harmony: int = 10,
    copy: bool = False,
    verbose: bool = True,
) -> Optional[AnnData]:
    """
    Integrating gene expression and spatial location to identify spatial domains via Scanpy and Squidpy.
    Original Code Repository: https://github.com/theislab/scanpy
    Original Code Repository: https://github.com/theislab/squidpy

    Args:
        adata: An Anndata object after normalization.
        spatial_key: the key in `.obsm` that corresponds to the spatial coordinate of each bucket.
        key_added: adata.obs key under which to add the cluster labels.
        n_pca_components: Number of principal components to compute.
                          If `n_pca_components` == None, the value at the inflection point of the PCA curve is
                          automatically calculated as n_comps.
        cluster_method:
        resolution: Resolution in the Louvain's Clustering method or Leiden's Clustering method.
        e_neigh: Number of nearest neighbor in gene expression space.
        s_neigh: Number of nearest neighbor in physical space.
        debatch: Whether to remove batch effects. This function is used in integrated analysis.
        batch_key: The name of the column in ``adata.obs`` that differentiates among experiments/batches.
                   Used when `debatch`== True.
        max_iter_harmony: Maximum number of rounds to run Harmony. One round of Harmony involves one clustering and one correction step.
                          Used when `debatch`== True.
        copy: Whether to copy `adata` or modify it inplace.
        verbose: Print information about clustering.

    Returns:
        Updates adata with the field ``adata.obs[key_added]``, containing the cluster result based on scanpy.

    Examples:
        >> import anndata as ad
        >> from cluster import qc_scanpy, sctransform, cluster_scsq
        >> adata = ad.read("/media/yao/Elements SE/BGI_Paper/spateo/drosophila_E16_18_a/align_bin20/E16-18h_a_S06.h5ad")
        >> qc_scanpy(adata=adata, mt_counts_threshold=5, save_qc_img="qc.png")
        >> sctransform(adata=adata, save_sct_img_1="sct1.png", save_sct_img_2="sct2.png")
        >> cluster_scsq(adata=adata, resolution=2.0)
    """

    adata = adata.copy() if copy else adata
    # Run PCA
    if n_pca_components is None:
        n_pca_components, _ = compute_pca_components(adata.X, save_curve_img=None)
    sc.pp.pca(adata, n_comps=n_pca_components)
    pca_key = "X_pca"
    # Remove batch effects.
    if debatch is True:
        harmony_debatch(
            adata,
            batch_key,
            basis="X_pca",
            adjusted_basis="X_pca_harmony",
            max_iter_harmony=max_iter_harmony,
        )
        pca_key = "X_pca_harmony"

    # Calculate the adjacent matrix.
    adj = spatial_adj_scsq(
        adata=adata,
        spatial_key=spatial_key,
        pca_key=pca_key,
        e_neigh=e_neigh,
        s_neigh=s_neigh,
    )

    # Run cluster.
    if cluster_method is "leiden":
        # Leiden's Clustering method.
        sc.tl.leiden(adata, adjacency=adj, resolution=resolution, key_added=key_added)
    elif cluster_method is "louvain":
        # Louvain's Clustering method.
        sc.tl.louvain(adata, adjacency=adj, resolution=resolution, key_added=key_added)

    return adata if copy else None
