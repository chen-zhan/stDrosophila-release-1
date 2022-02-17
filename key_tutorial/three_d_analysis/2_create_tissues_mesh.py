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


# tissues (fat body)
fb_adata = adata[adata.obs["anno"] == "fat body", :].copy()
fb_adata = sd.tl.om_kde(fb_adata, threshold=0.2)
fb_adata = sd.tl.om_EllipticEnvelope(fb_adata, threshold=0.2)
fb_mesh = sd.tl.build_three_d_model(
    adata=fb_adata,
    coordsby="spatial",
    groupby="anno",
    group_show=["fat body"],
    group_cmap=["#F56867"],
    group_amap=0,
    surf_color="#F56867",
    surf_alpha=0.5,
    cs_method="alpha_shape",
    cs_method_args={"alpha": 5},
    smoothing=True,
    n_surf=10000,
    voxelize=True,
    voxel_size=[0.5, 0.5, 0.7],
    voxel_smooth=300,
)
sd.pl.easy_three_d_plot(
    mesh=fb_mesh,
    scalar="groups",
    ambient=0.3,
    opacity=0.5,
    save=r"D:\BGIpy37_pytorch113\three_d_analysis_image\create_tissues_mesh\only_fat_body_no.png",
    off_screen=False,
    cpos="iso",
    legend_loc="lower right",
    legend_size=(0.1, 0.2),
)


# tissues (CNS)
cns_adata = adata[adata.obs["anno"] == "CNS", :].copy()
cns_adata = sd.tl.om_kde(cns_adata, threshold=0.2)
cns_adata = sd.tl.om_EllipticEnvelope(cns_adata, threshold=0.2)
cns_mesh = sd.tl.build_three_d_model(
    adata=cns_adata,
    coordsby="spatial",
    groupby="anno",
    group_show=["CNS"],
    group_cmap=["#FEB915"],
    group_amap=0,
    surf_color="#FEB915",
    surf_alpha=0.5,
    cs_method="alpha_shape",
    cs_method_args={"alpha": 5},
    smoothing=True,
    n_surf=10000,
    voxelize=True,
    voxel_size=[0.5, 0.5, 0.7],
    voxel_smooth=300,
)
sd.pl.easy_three_d_plot(
    mesh=cns_mesh,
    scalar="groups",
    ambient=0.3,
    opacity=0.5,
    save=r"D:\BGIpy37_pytorch113\three_d_analysis_image\create_tissues_mesh\only_CNS_no.png",
    off_screen=False,
    cpos="iso",
    legend_loc="lower right",
    legend_size=(0.1, 0.2),
)


# tissues (body + CNS + fat body)
body_mesh = sd.tl.build_three_d_model(
    adata=adata,
    coordsby="spatial",
    groupby=None,
    group_amap=0,
    surf_color="gainsboro",
    surf_alpha=0.5,
    cs_method="basic",
    cs_method_args=None,
    smoothing=True,
    n_surf=10000,
    voxelize=True,
    voxel_size=[0.5, 0.5, 0.7],
    voxel_smooth=300,
)
import pyvista as pv

p = pv.Plotter()
p.add_mesh(body_mesh, rgba=True, scalars="groups_rgba", ambient=0.3, opacity=0.5)
p.add_mesh(fb_mesh, rgba=True, scalars="groups_rgba", ambient=0.3, opacity=0.5)
p.add_mesh(cns_mesh, rgba=True, scalars="groups_rgba", ambient=0.3, opacity=0.5)
p.show(
    screenshot=r"D:\BGIpy37_pytorch113\three_d_analysis_image\create_tissues_mesh\combination.png"
)
