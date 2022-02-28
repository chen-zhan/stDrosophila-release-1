import anndata as ad
import numpy as np
import stDrosophila as sd
import warnings

warnings.filterwarnings("ignore")

# Example data
file = r"D:\BGIpy37_pytorch113\E16-18_a_SCT_anno.h5ad"
adata = ad.read(file)
adata.obs["x"] = adata.obs["x"] - adata.obs["x"].min()
adata.obs["y"] = adata.obs["y"] - adata.obs["y"].min()
adata.obs["z"] = adata.obs["z"] - 4.9
adata.obs[["x", "y", "z"]] = adata.obs[["x", "y", "z"]].round(2)
adata.obsm["spatial"] = adata.obs[["x", "y", "z"]].values

# tissues (fat body)
fb_adata = adata[adata.obs["anno"] == "fat body", :].copy()
fb_adata = sd.tl.om_kde(fb_adata, threshold=0.4)
fb_adata = sd.tl.om_EllipticEnvelope(fb_adata, threshold=0.2)
fb_pcd, fb_surf = sd.tl.construct_three_d_mesh(
    adata=fb_adata, coordsby="spatial", groupby="Adh", key_added="groups",
    mask=None, mesh_style="surf", mesh_color="orange", mesh_alpha=0.5,
    pcd_cmap="hot_r", pcd_amap=1.0, pcd_voxelize=False, pcd_voxel_size=[0.5, 0.5, 0.5],
    cs_method="alpha_shape", cs_method_args={"al_alpha": 10}, surf_smoothness=200, n_surf=50000, vol_smoothness=200,
)

sd.pl.three_d_plot(
    mesh=fb_pcd, key="groups", off_screen=False, window_size=(1024, 768),  background="white", background_r="black",
    ambient=0.3, opacity=1.0, initial_cpo="iso", legend_loc="lower right", legend_size=(0.1, 0.1),
    filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_img\fb_pcd.png",
    view_up=(0.5, 0.5, 1), framerate=15, plotter_filename=None
)

sd.pl.three_d_plot(
    mesh=fb_surf, key="groups", off_screen=False, window_size=(1024, 768), background="white", background_r="black",
    ambient=0.3, opacity=1.0, initial_cpo="iso", legend_loc="lower right", legend_size=(0.1, 0.1),
    filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_img\fb_surf.png",
    view_up=(0.5, 0.5, 1), framerate=15, plotter_filename=None
)

sd.tl.mesh_to_vtk(mesh=fb_pcd, filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\fb_pcd.vtk")
sd.tl.mesh_to_vtk(mesh=fb_surf, filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\fb_surf.vtk")


# tissues (CNS)
cns_adata = adata[adata.obs["anno"] == "CNS", :].copy()
cns_adata = sd.tl.om_kde(cns_adata, threshold=0.4)
cns_adata = sd.tl.om_EllipticEnvelope(cns_adata, threshold=0.2)

cns_pcd, cns_surf = sd.tl.construct_three_d_mesh(
    adata=cns_adata, coordsby="spatial", groupby="miple1", key_added="groups",
    mask=None, mesh_style="surf", mesh_color="skyblue", mesh_alpha=0.5,
    pcd_cmap="hot_r", pcd_amap=1.0, pcd_voxelize=False, pcd_voxel_size=[0.5, 0.5, 0.5],
    cs_method="alpha_shape", cs_method_args={"al_alpha": 10}, surf_smoothness=200, n_surf=50000, vol_smoothness=200,
)

sd.pl.three_d_plot(
    mesh=cns_pcd, key="groups", off_screen=False, window_size=(1024, 768), background="white", background_r="black",
    ambient=0.3, opacity=1.0, initial_cpo="iso", legend_loc="lower right", legend_size=(0.1, 0.1),
    filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_img\cns_pcd.png",
    view_up=(0.5, 0.5, 1), framerate=15, plotter_filename=None
)

sd.pl.three_d_plot(
    mesh=cns_surf, key="groups", off_screen=False, window_size=(1024, 768), background="white", background_r="black",
    ambient=0.3, opacity=1.0, initial_cpo="iso", legend_loc="lower right", legend_size=(0.1, 0.1),
    filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_img\cns_surf.png",
    view_up=(0.5, 0.5, 1), framerate=15, plotter_filename=None
)

sd.tl.mesh_to_vtk(mesh=cns_pcd, filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\cns_pcd.vtk")
sd.tl.mesh_to_vtk(mesh=cns_surf, filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\cns_surf.vtk")
