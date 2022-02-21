import anndata as ad
import numpy as np
import stDrosophila as sd


# Example data
file = "/three_d_reconstruction/example_data/E16-18_a_SCT_anno.h5ad"
adata = ad.read(file)
adata.obs["x"] = adata.obs["x"] - adata.obs["x"].min()
adata.obs["y"] = adata.obs["y"] - adata.obs["y"].min()
adata.obs["z"] = adata.obs["z"] - 4.9
adata.obs[["x", "y", "z"]] = adata.obs[["x", "y", "z"]].round(2)
adata.obsm["spatial"] = adata.obs[["x", "y", "z"]].values


# whole body (basic)
pcd, volume = sd.tl.construct_three_d_mesh(
    adata=adata,
    coordsby="spatial",
    groupby="anno",
    group_show="all",
    group_cmap="rainbow",
    group_amap=1.0,
    gene_show="all",
    gene_cmap="hot_r",
    gene_amap=1.0,
    mask_color="gainsboro",
    mask_alpha=0,
    cs_method="basic",
    cs_method_args=None,
    surf_smoothness=100,
    n_surf=10000,
    vol_color="gainsboro",
    vol_alpha=0.5,
    vol_smoothness=200,
    pcd_voxelize=True,
    pcd_voxel_size=[0.5, 0.5, 0.7],
    coodtype=np.float64,
    expdtype=np.float64,
)
complete_mesh_1 = sd.tl.merge_mesh([pcd, volume])
sd.pl.easy_three_d_plot(
    mesh=complete_mesh_1,
    scalar="groups",
    ambient=0.3,
    opacity=0.5,
    save=r"/home/yao/BGIpy37_pytorch113/three_d_reconstruction/create_whole_mesh/basic_whole_body.png",
    off_screen=True,
    cpos="iso",
    legend_loc="lower right",
    legend_size=(0.1, 0.2),
)