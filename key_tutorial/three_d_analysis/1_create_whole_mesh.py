import anndata as ad
import numpy as np
import stDrosophila as sd


# Example data
file = "D:\BGIpy37_pytorch113\E16-18_a_SCT_anno.h5ad"
adata = ad.read(file)
adata.obs["x"] = adata.obs["x"] - adata.obs["x"].min()
adata.obs["y"] = adata.obs["y"] - adata.obs["y"].min()
adata.obs["z"] = adata.obs["z"] - 4.9
adata.obs[["x", "y", "z"]] = adata.obs[["x", "y", "z"]].round(2)
adata.obsm["spatial"] = adata.obs[["x", "y", "z"]].values


# whole body (basic)
mesh1 = sd.tl.build_three_d_model(
    adata=adata, coordsby="spatial", groupby="anno",                # basic info
    group_show="all", group_cmap="rainbow", group_amap=0.5,         # group set
    gene_show="all", gene_cmap="hot_r", gene_amap=0.5,              # gene expression set
    mask_color="gainsboro", mask_alpha=0,                           # mask set (not displayed)
    surf_color="gainsboro", surf_alpha=0.5, cs_method="basic", cs_method_args=None, smoothing=True, n_surf=10000,   # surface set
    voxelize=True, voxel_size=[0.5, 0.5, 0.7], voxel_smooth=300,    # voxelize set
    coodtype=np.float64, expdtype=np.float64                        # data type
)
sd.pl.easy_three_d_plot(mesh=mesh1, scalar="groups", ambient=0.3, opacity=0.5,
                        save=r"D:\BGIpy37_pytorch113\three_d_analysis_image\create_whole_mesh\basic_whole_body.png",
                        off_screen=True, cpos="iso", legend_loc="lower right", legend_size=(0.1, 0.2))


# whole body (one tissue)
mesh2 = sd.tl.build_three_d_model(
    adata=adata, coordsby="spatial", groupby="anno",                # basic info
    group_show=["fat body"], group_cmap="rainbow", group_amap=0.5,  # group set
    gene_show="all", gene_cmap="hot_r", gene_amap=0.5,              # gene expression set
    mask_color="gainsboro", mask_alpha=0,                           # mask set (not displayed)
    surf_color="gainsboro", surf_alpha=0.5, cs_method="basic", cs_method_args=None, smoothing=True, n_surf=10000,   # surface set
    voxelize=True, voxel_size=[0.5, 0.5, 0.7], voxel_smooth=300,    # voxelize set
    coodtype=np.float64, expdtype=np.float64                        # data type
)
sd.pl.easy_three_d_plot(mesh=mesh2, scalar="groups", ambient=0.3, opacity=0.5,
                        save=r"D:\BGIpy37_pytorch113\three_d_analysis_image\create_whole_mesh\basic_one_tissue.png",
                        off_screen=True, cpos="iso", legend_loc="lower right", legend_size=(0.1, 0.2))


# whole body (only surface)
mesh3 = sd.tl.build_three_d_model(
    adata=adata, coordsby="spatial", groupby=None,                # basic info
    group_show="all", group_cmap="rainbow", group_amap=0,  # group set
    gene_show="all", gene_cmap="hot_r", gene_amap=0.5,              # gene expression set
    mask_color="gainsboro", mask_alpha=0,                           # mask set (not displayed)
    surf_color="gainsboro", surf_alpha=0.5, cs_method="basic", cs_method_args=None, smoothing=True, n_surf=10000,   # surface set
    voxelize=True, voxel_size=[0.5, 0.5, 0.7], voxel_smooth=300,    # voxelize set
    coodtype=np.float64, expdtype=np.float64                        # data type
)
sd.pl.easy_three_d_plot(mesh=mesh3, scalar="groups", ambient=0.3, opacity=0.5,
                        save=r"D:\BGIpy37_pytorch113\three_d_analysis_image\create_whole_mesh\basic_only_surface.png",
                        off_screen=False, cpos="iso", legend_loc="lower right", legend_size=(0.1, 0.2))


# whole body (one gene)
mesh4 = sd.tl.build_three_d_model(
    adata=adata, coordsby="spatial", groupby="anno",                # basic info
    group_show="all", group_cmap="rainbow", group_amap=0.5,         # group set
    gene_show=["Adh"], gene_cmap="hot_r", gene_amap=0.5,              # gene expression set
    mask_color="gainsboro", mask_alpha=0,                           # mask set (not displayed)
    surf_color="gainsboro", surf_alpha=0.5, cs_method="basic", cs_method_args=None, smoothing=True, n_surf=10000,   # surface set
    voxelize=True, voxel_size=[0.5, 0.5, 0.7], voxel_smooth=300,    # voxelize set
    coodtype=np.float64, expdtype=np.float64                        # data type
)
sd.pl.easy_three_d_plot(mesh=mesh4, scalar="genes", ambient=0.3, opacity=0.5,
                        save=r"D:\BGIpy37_pytorch113\three_d_analysis_image\create_whole_mesh\basic_one_gene.png",
                        off_screen=True, cpos="iso", legend_loc="lower right", legend_size=(0.1, 0.2))


# whole body (one gene)
mesh5 = sd.tl.build_three_d_model(
    adata=adata, coordsby="spatial", groupby="anno",                    # basic info
    group_show=["fat body"], group_cmap="rainbow", group_amap=0.5,      # group set
    gene_show=["Adh"], gene_cmap="hot_r", gene_amap=0.5,                # gene expression set
    mask_color="gainsboro", mask_alpha=0,                               # mask set (not displayed)
    surf_color="gainsboro", surf_alpha=0.5, cs_method="basic", cs_method_args=None, smoothing=True, n_surf=10000,   # surface set
    voxelize=True, voxel_size=[0.5, 0.5, 0.7], voxel_smooth=300,        # voxelize set
    coodtype=np.float64, expdtype=np.float64                            # data type
)
sd.pl.easy_three_d_plot(mesh=mesh5, scalar="genes", ambient=0.3, opacity=0.5,
                        save=r"D:\BGIpy37_pytorch113\three_d_analysis_image\create_whole_mesh\basic_one_gene_one_tissue.png",
                        off_screen=True, cpos="iso", legend_loc="lower right", legend_size=(0.1, 0.2))


