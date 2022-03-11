import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import seaborn as sns
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA

def _SpaGCN_Cluster(adata, n_comps=50, n_neighbors=10, x_coordinate='new_x', y_coordinate='new_y',
                    p=0.5, res_st=0.4, n_clusters=10, n_seed=100,numItermaxSpa=200):

    # Set coordinates
    x_coo = adata.obs[x_coordinate].tolist()
    y_coo = adata.obs[y_coordinate].tolist()
    # Calculate the adjacent matrix
    adj = spg.calculate_adj_matrix(x=x_coo, y=y_coo, histology=False)
    # Search for suitable resolution
    l = spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    r_seed = t_seed = n_seed
    res = spg.search_res(adata, adj, l, n_clusters, start=res_st, step=0.1, tol=5e-3, lr=0.05, max_epochs=200,
                         r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
    clf = spg.SpaGCN()
    clf.set_l(l)
    # Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    # Run
    clf.train(adata, adj, num_pcs=n_comps, n_neighbors=n_neighbors, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05,max_epochs=numItermaxSpa)
    y_pred, prob = clf.predict()
    adata.obs["pred"] = y_pred
    adata.obs["pred"] = adata.obs["pred"].astype('category')
    adj_2d = spg.calculate_adj_matrix(x=x_coo, y=y_coo, histology=False)
    refined_pred = spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d,shape="hexagon")
    adata.obs["refined_pred"] = refined_pred
    adata.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
    # Set colors used
    plot_color = ["#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#D1D1D1", "#6D1A9C", "#15821E", "#3A84E6",
                  "#997273", "#787878", "#DB4C6C", "#9E7A7A", "#554236", "#AF5F3C", "#93796C", "#F9BD3F", "#DAB370",
                  "#877F6C", "#268785"]
    adata.uns["pred_colors"] = list(plot_color[:len(adata.obs["pred"].unique())])
    adata.uns["refined_pred_colors"] = list(plot_color[:len(adata.obs["refined_pred"].unique())])
    return adata


def _SpaGCN_Cluster_Plot(AdataList=None, opath=None, spot_size=None):
    fig = plt.figure(figsize=[14, int(len(AdataList)) * 7], dpi=100)
    sns.set(style="white")
    for num, adata in enumerate(AdataList):
        ax1 = plt.subplot(int(len(AdataList)), 2, num * 2 + 1)
        palette1 = sns.color_palette(adata.uns['pred_colors'])
        sc.pl.spatial(adata, spot_size=spot_size, color='pred', palette=palette1, show=False, ax=ax1)
        ax1.set_title(f"{adata.obs['slice_ID'][0]}_pred")

        ax2 = plt.subplot(int(len(AdataList)), 2, num * 2 + 2)
        palette2 = sns.color_palette(adata.uns['refined_pred_colors'])
        sc.pl.spatial(adata, spot_size=spot_size, color='refined_pred', palette=palette2, show=False, ax=ax2)
        ax2.set_title(f"{adata.obs['slice_ID'][0]}_refined_pred")
    plt.tight_layout()
    plt.savefig(os.path.join(opath, 'refined_pred.png'), dpi=100)
    plt.close(fig)


def _SpaGCN_SVGs(raw, adata, x_coordinate,y_coordinate, target_cluster, min_in_group_fraction, min_in_out_group_ratio, min_fold_change, find_neighbor_clusters_ratio):
    # Search radius such that each spot in the target domain has approximately 10 neighbors on average
    x_coo = adata.obs[x_coordinate].tolist()
    y_coo = adata.obs[y_coordinate].tolist()
    adj_2d = spg.calculate_adj_matrix(x=x_coo, y=y_coo, histology=False)
    start, end = np.quantile(adj_2d[adj_2d != 0], q=0.001), np.quantile(adj_2d[adj_2d != 0], q=0.1)
    r = spg.search_radius(target_cluster=target_cluster, cell_id=adata.obs.index.tolist(), x=x_coo, y=y_coo,
                          pred=adata.obs["refined_pred"].tolist(), start=start, end=end, num_min=10, num_max=14, max_run=100)
    # Detect neighboring domains
    nbr_domians = spg.find_neighbor_clusters(target_cluster=target_cluster,
                                             cell_id=raw.obs.index.tolist(),
                                             x=raw.obs[x_coordinate].tolist(),
                                             y=raw.obs[y_coordinate].tolist(),
                                             pred=raw.obs["refined_pred"].tolist(),
                                             radius=r,
                                             ratio=find_neighbor_clusters_ratio)
    nbr_domians = nbr_domians[0:3]
    de_genes_info = spg.rank_genes_groups(input_adata=raw,
                                          target_cluster=target_cluster,
                                          nbr_list=nbr_domians,
                                          label_col="refined_pred",
                                          adj_nbr=True,
                                          log=True)

    # Filter genes
    de_genes_info = de_genes_info[(de_genes_info["pvals_adj"] < 0.05)]
    filtered_info = de_genes_info
    filtered_info = filtered_info[(filtered_info["pvals_adj"] < 0.05) &
                                  (filtered_info["in_out_group_ratio"] > min_in_out_group_ratio) &
                                  (filtered_info["in_group_fraction"] > min_in_group_fraction) &
                                  (filtered_info["fold_change"] > min_fold_change)]
    filtered_info = filtered_info.sort_values(by="in_group_fraction", ascending=False)
    filtered_info["target_dmain"] = target_cluster
    filtered_info["neighbors"] = str(nbr_domians)
    print("SVGs for domain ", str(target_cluster), ":", filtered_info["genes"].tolist())

    return filtered_info


def _SpaGCN_SVGs_Plot(raw, filtered_info, opath, x_coordinate, y_coordinate, target_cluster):
    # Plot refinedspatial domains
    color_self = clr.LinearSegmentedColormap.from_list('pink_green', ['#3AB370', "#EAE7CC", "#FD1593"], N=256)
    svgs_list = filtered_info["genes"].tolist()
    sns.set(style="white")
    for num, g in enumerate(svgs_list):
        raw.obs["exp"] = raw.X[:, raw.var.index == g]
        fig = plt.figure(figsize=[7, 7], dpi=100)
        sns.set(style="white")
        ax = sc.pl.scatter(raw, alpha=1, x=x_coordinate, y=y_coordinate, color="exp", title=g, color_map=color_self,
                      show=False, size=100000 / raw.shape[0])
        ax.set_aspect('equal', 'box')
        ax.axes.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(opath, f"SVGs_{target_cluster}_{g}.png"), dpi=100)
        plt.clf()
        plt.close(fig)


def _SpaGCN_MetaGenes(raw, opath, target_cluster,start_gene,x_coordinate,y_coordinate):
    meta_name, meta_exp = spg.find_meta_gene(input_adata=raw,
                                             pred=raw.obs["refined_pred"].tolist(),
                                             target_domain=target_cluster,
                                             start_gene=start_gene,
                                             mean_diff=0,
                                             early_stop=True,
                                             max_iter=100,
                                             use_raw=False)
    raw.obs["meta"] = meta_exp
    color_self = clr.LinearSegmentedColormap.from_list('pink_green', ['#3AB370', "#EAE7CC", "#FD1593"], N=256)
    ax = sc.pl.scatter(raw, alpha=1, x=x_coordinate, y=y_coordinate, color="meta", title=meta_name, color_map=color_self,
                       show=False, size=100000 / raw.shape[0])
    ax.set_aspect('equal', 'box')
    ax.axes.invert_yaxis()
    plt.savefig(os.path.join(opath, f'meta_gene_{target_cluster}.png'), dpi=100)
    plt.close()
    return meta_name,raw

def _SpaGCN_Genes(reg_adata,adata,opath,x_coordinate,y_coordinate):
    # Read in raw data
    raw = reg_adata
    raw.var_names_make_unique()
    raw.obs["refined_pred"] = adata.obs["refined_pred"].astype('category')
    # Convert sparse matrix to non-sparse
    raw.X = (raw.X.A if issparse(raw.X) else raw.X)
    raw.raw = raw
    sc.pp.log1p(raw)

    min_in_group_fraction = 0.8
    min_in_out_group_ratio = 0.8
    min_fold_change = 1.0
    find_neighbor_clusters_ratio = 0.3

    cluster_list = raw.obs["refined_pred"].unique().tolist()
    cluster_list.sort()
    svg_info = pd.DataFrame()
    for target_cluster in cluster_list:
        raw_cycle = raw
        cluster_path = os.path.join(opath,f'cluster_{target_cluster}')
        svgs_image_path = os.path.join(cluster_path,'SVGs_images')
        if not os.path.exists(cluster_path):
            os.mkdir(cluster_path)
        if not os.path.exists(svgs_image_path):
            os.mkdir(svgs_image_path)
        filtered_info = _SpaGCN_SVGs(raw=raw_cycle, adata=adata,x_coordinate=x_coordinate,
                                     y_coordinate=y_coordinate, target_cluster=target_cluster,
                                     min_in_group_fraction=min_in_group_fraction,
                                     min_in_out_group_ratio=min_in_out_group_ratio,
                                     min_fold_change=min_fold_change,
                                     find_neighbor_clusters_ratio=find_neighbor_clusters_ratio)
        filtered_info = filtered_info.sort_values(by='fold_change',ascending=False,na_position='last')
        filtered_info.index= range(len(filtered_info.index))
        if len(filtered_info.index) != 0:
            svg_info = pd.concat([svg_info, filtered_info], axis=0)
            _SpaGCN_SVGs_Plot(raw=raw_cycle, filtered_info=filtered_info, opath= svgs_image_path,
                              x_coordinate=x_coordinate, y_coordinate=y_coordinate, target_cluster=target_cluster)

            meta_name,raw = _SpaGCN_MetaGenes(raw=raw_cycle, opath=cluster_path, target_cluster=target_cluster,
                                start_gene=filtered_info['genes'][0],x_coordinate=x_coordinate,y_coordinate=y_coordinate)
            print(meta_name)
    svg_info.to_csv(os.path.join(opath, 'svgs.csv'), index=False)
    return svg_info

ipath = 'D:\BGI\ST_Drosophila\Test_data\E16_18_d_bin20_Alignment\h5ad'
pathList = [os.path.join(root, filename) for root, dirs, files in os.walk(ipath) for filename in files]
pathList.sort()
slicesList = [sc.read_h5ad(i) for i in pathList]
opath = 'D:\BGI\ST_Drosophila\Test_data\E16_18_d_bin20_Cluster_SpaGCN'
h5adOpath = os.path.join(opath, 'h5ad')
if not os.path.exists(opath):
    os.mkdir(opath)
if not os.path.exists(h5adOpath):
    os.mkdir(h5adOpath)

cluSliceList = []
for slice in slicesList:
    # QC
    spg.prefilter_genes(slice, min_cells=3)
    spg.prefilter_specialgenes(slice)
    # 标准化
    raw_adata = slice.copy()
    sc.pp.normalize_per_cell(slice)
    sc.pp.log1p(slice)
    # 聚类(leiden)
    adata = _SpaGCN_Cluster(adata=slice, n_comps=50, n_neighbors=10, x_coordinate='x', y_coordinate='y',
                            p=0.5, res_st=0.4, n_clusters=7, n_seed=100, numItermaxSpa=200)

    '''
    sub_path = os.path.join(opath, adata.obs['slice_ID'][0])
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)
    _SpaGCN_Genes(reg_adata=raw_adata, adata=adata,opath=sub_path, x_coordinate='x', y_coordinate='y')'''
    sc.tl.rank_genes_groups(adata, 'refined_pred', method="t-test", key_added=f'refined_pred_rank_genes_groups')
    cluSliceList.append(adata)

# 输出聚类结果
h5adOpath = os.path.join(opath, 'h5ad')
if not os.path.exists(h5adOpath):
    os.mkdir(h5adOpath)
for adata in cluSliceList:
    adata.write_h5ad(os.path.join(h5adOpath, f"{adata.obs['slice_ID'][0]}.h5ad"))
# 聚类结果可视化
_SpaGCN_Cluster_Plot(AdataList=cluSliceList, opath=opath, spot_size=1)