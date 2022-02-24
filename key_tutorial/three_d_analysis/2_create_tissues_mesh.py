import anndata as ad
import numpy as np
import stDrosophila as sd
import warnings

warnings.filterwarnings("ignore")

# Example data
file = "/home/yao/BGIpy37_pytorch113/three_d_reconstruction/example_data/E16-18_a_SCT_anno.h5ad"
adata = ad.read(file)
adata.obs["x"] = adata.obs["x"] - adata.obs["x"].min()
adata.obs["y"] = adata.obs["y"] - adata.obs["y"].min()
adata.obs["z"] = adata.obs["z"] - 4.9
adata.obs[["x", "y", "z"]] = adata.obs[["x", "y", "z"]].round(2)
adata.obsm["spatial"] = adata.obs[["x", "y", "z"]].values
print(adata.obs["anno"].unique().tolist())
# tissues (salivary gland)
sg_adata = adata[adata.obs["anno"] == "midgut", :].copy()
sg_adata = sd.tl.om_kde(sg_adata, threshold=0.4)
sg_adata = sd.tl.om_EllipticEnvelope(sg_adata, threshold=0.4)
sg_pcd, sg_volume = sd.tl.construct_three_d_mesh(
    adata=sg_adata,
    coordsby="spatial",
    cs_method="alpha_shape",
    cs_method_args={"al_alpha": 10},
    surf_smoothness=100,
    n_surf=10000,
    vol_color="yellowgreen",
    vol_alpha=1,
    vol_smoothness=500,
)
sd.pl.easy_three_d_plot(
    mesh=sg_volume,
    scalar="groups",
    ambient=0.3,
    opacity=1,
    save=r"/home/yao/BGIpy37_pytorch113/three_d_reconstruction/create_tissues_mesh/tissue_salivarygland.png",
    off_screen=True,
    legend_size=(0.1, 0.2),
)

# tissues (trachea)
trachea_adata = adata[adata.obs["anno"] == "trachea", :].copy()
trachea_adata = sd.tl.om_kde(trachea_adata, threshold=0.4)
trachea_adata = sd.tl.om_EllipticEnvelope(trachea_adata, threshold=0.2)
trachea_pcd, trachea_volume = sd.tl.construct_three_d_mesh(
    adata=trachea_adata,
    coordsby="spatial",
    cs_method="alpha_shape",
    cs_method_args={"al_alpha": 13},
    surf_smoothness=100,
    n_surf=10000,
    vol_color="skyblue",
    vol_alpha=1,
    vol_smoothness=500,
)
sd.pl.easy_three_d_plot(
    mesh=trachea_volume,
    scalar="groups",
    ambient=0.3,
    opacity=1,
    save=r"/home/yao/BGIpy37_pytorch113/three_d_reconstruction/create_tissues_mesh/tissue_trachea.png",
    off_screen=True,
    legend_size=(0.1, 0.2),
)

# tissues (muscle)
muscle_adata = adata[adata.obs["anno"] == "muscle", :].copy()
muscle_adata = sd.tl.om_kde(muscle_adata, threshold=0.4)
muscle_adata = sd.tl.om_EllipticEnvelope(muscle_adata, threshold=0.4)
muscle_pcd, muscle_volume = sd.tl.construct_three_d_mesh(
    adata=muscle_adata,
    coordsby="spatial",
    cs_method="alpha_shape",
    cs_method_args={"al_alpha": 10},
    surf_smoothness=100,
    n_surf=10000,
    vol_color="yellow",
    vol_alpha=1,
    vol_smoothness=500,
)
sd.pl.easy_three_d_plot(
    mesh=muscle_volume,
    scalar="groups",
    ambient=0.3,
    opacity=1,
    save=r"/home/yao/BGIpy37_pytorch113/three_d_reconstruction/create_tissues_mesh/tissue_muscle.png",
    off_screen=True,
    legend_size=(0.1, 0.2),
)

# tissues (fat body)
fb_adata = adata[adata.obs["anno"] == "fat body", :].copy()
fb_adata = sd.tl.om_kde(fb_adata, threshold=0.4)
fb_adata = sd.tl.om_EllipticEnvelope(fb_adata, threshold=0.2)
fb_pcd, fb_volume = sd.tl.construct_three_d_mesh(
    adata=fb_adata,
    coordsby="spatial",
    cs_method="alpha_shape",
    cs_method_args={"al_alpha": 10},
    surf_smoothness=100,
    n_surf=10000,
    vol_color="red",
    vol_alpha=1,
    vol_smoothness=500,
)
sd.pl.easy_three_d_plot(
    mesh=fb_volume,
    scalar="groups",
    ambient=0.3,
    opacity=1,
    save=r"/home/yao/BGIpy37_pytorch113/three_d_reconstruction/create_tissues_mesh/tissue_fatbody.png",
    off_screen=True,
    legend_size=(0.1, 0.2),
)

# tissues (CNS)
cns_adata = adata[adata.obs["anno"] == "CNS", :].copy()
cns_adata = sd.tl.om_kde(cns_adata, threshold=0.4)
cns_adata = sd.tl.om_EllipticEnvelope(cns_adata, threshold=0.2)
cns_pcd, cns_volume = sd.tl.construct_three_d_mesh(
    adata=cns_adata,
    coordsby="spatial",
    cs_method="alpha_shape",
    cs_method_args={"al_alpha": 10},
    surf_smoothness=100,
    n_surf=10000,
    vol_color="blue",
    vol_alpha=1,
    vol_smoothness=500,
)
sd.pl.easy_three_d_plot(
    mesh=cns_volume,
    scalar="groups",
    ambient=0.3,
    opacity=1,
    save=r"/home/yao/BGIpy37_pytorch113/three_d_reconstruction/create_tissues_mesh/tissue_CNS.png",
    off_screen=True,
    legend_size=(0.1, 0.2),
)

# whole body
body_pcd, body_volume = sd.tl.construct_three_d_mesh(
    adata=adata,
    coordsby="spatial",
    cs_method="basic",
    cs_method_args=None,
    surf_smoothness=100,
    n_surf=50000,
    vol_color="gainsboro",
    vol_alpha=0.5,
    vol_smoothness=500,
)

complete_mesh = sd.tl.merge_mesh(
    [sg_volume, trachea_volume, muscle_volume, fb_volume, cns_volume, body_volume]
)
for cpo in ["xy", "xz", "yz", "yx", "zx", "zy", "iso"]:
    sd.pl.easy_three_d_plot(
        mesh=complete_mesh,
        scalar="groups",
        ambient=0.3,
        opacity=1,
        save=rf"/home/yao/BGIpy37_pytorch113/three_d_reconstruction/create_tissues_mesh/whole_{cpo}.png",
        cpos=cpo,
        off_screen=True,
        legend_size=(0.1, 0.2),
    )
