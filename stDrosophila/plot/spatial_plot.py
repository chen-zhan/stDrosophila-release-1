import math

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns

from anndata import AnnData
from typing import Optional, Union, Sequence


def spatial_plot(adata: Union[AnnData, Sequence[AnnData]],
                 cluster_col: Optional[str] = None,
                 save: Optional[str] = None,
                 slice_col: Optional[str] = "slice",
                 spot_size: Optional[int] = 1,
                 ):

    plot_color = ["#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#D1D1D1", "#6D1A9C", "#15821E", "#3A84E6",
                  "#997273", "#787878", "#DB4C6C", "#9E7A7A", "#554236", "#AF5F3C", "#93796C", "#F9BD3F", "#DAB370",
                  "#877F6C", "#268785"]

    if isinstance(adata, list):
        adata_list = adata
    else:
        adata_ca = adata.obs[slice_col].unique().tolist()
        adata_list = [adata] if len(adata_ca) == 1 else [adata[adata.obs[slice_col] == i] for i in adata_ca]

    # Embedding
    ind_num = math.ceil(int(len(adata_list)) / 4)
    col_num = 4 if ind_num > 1 else len(adata_list)
    fig_sp = plt.figure(figsize=[col_num * 7, ind_num * 7], dpi=100)
    sns.set(style="white")
    if cluster_col is None:
        for num, sub_adata in enumerate(adata_list):
            ax = plt.subplot(ind_num, col_num, num + 1)
            sc.pl.spatial(sub_adata, spot_size=spot_size, show=False, ax=ax)
            ax.set_title(f"{sub_adata.obs[slice_col][0]}")
    else:
        cluster_ca = []
        for sub_adata in adata_list:
            cluster_ca.extend(sub_adata.obs[cluster_col].unique().tolist())
        cluster_ca = list(set(cluster_ca))
        cluster_ca.sort()
        cluster_color = {str(cluster): color for cluster, color in zip(cluster_ca, plot_color)}
        for num, sub_adata in enumerate(adata_list):
            ax = plt.subplot(ind_num, col_num, num + 1)
            sub_adata_cluster_ca = sub_adata.obs[cluster_col].unique().tolist()
            sub_adata_cluster_ca.sort()
            palette = sns.color_palette([cluster_color[str(i)] for i in sub_adata_cluster_ca])
            sc.pl.spatial(sub_adata, spot_size=spot_size, color=cluster_col, show=False, ax=ax, palette=palette)

            ax.set_title(f"{sub_adata.obs[slice_col][0]}")
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=100)
    plt.close(fig_sp)
    plt.clf()
