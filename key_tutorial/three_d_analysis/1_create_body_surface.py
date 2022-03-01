import anndata as ad
import stDrosophila as sd

# Example data
file = r"D:\BGIpy37_pytorch113\E16-18_a_SCT_anno.h5ad"
adata = ad.read(file)
adata.obs["x"] = adata.obs["x"] - adata.obs["x"].min()
adata.obs["y"] = adata.obs["y"] - adata.obs["y"].min()
adata.obs["z"] = adata.obs["z"] - 4.9
adata.obs[["x", "y", "z"]] = adata.obs[["x", "y", "z"]].round(2)
adata.obsm["spatial"] = adata.obs[["x", "y", "z"]].values
# print(adata.obs["anno"].unique().tolist())

# create pcd and surface
pcd, surf_mesh = sd.tl.construct_three_d_mesh(
    adata=adata,
    coordsby="spatial",
    groupby="anno",
    key_added="groups",
    mask=None,
    mesh_style="surf",
    mesh_color="gainsboro",
    mesh_alpha=0.5,
    pcd_cmap="rainbow",
    pcd_amap=0.5,
    pcd_voxelize=False,
    pcd_voxel_size=[0.5, 0.5, 0.5],
    cs_method="basic",
    cs_method_args=None,
    surf_smoothness=500,
    n_surf=50000,
    vol_smoothness=200,
)

sd.pl.three_d_plot(
    mesh=surf_mesh,
    key="groups",
    off_screen=False,
    window_size=(1024, 768),
    background="black",
    background_r="white",
    ambient=0.3,
    opacity=1.0,
    initial_cpo="iso",
    legend_loc="lower right",
    legend_size=(0.1, 0.1),
    filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_img\basic_surf.png",
    view_up=(0.5, 0.5, 1),
    framerate=15,
    plotter_filename=None,
)

sd.tl.mesh_to_vtk(
    mesh=surf_mesh,
    filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\basic_surf.vtk",
)
