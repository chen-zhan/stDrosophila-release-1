import anndata as ad
import numpy as np
import stDrosophila as sd
from pyvista import MultiBlock

# Example data
file = r"D:\BGIpy37_pytorch113\E16-18_a_SCT_anno.h5ad"
adata = ad.read(file)
adata.obs["x"] = adata.obs["x"] - adata.obs["x"].min()
adata.obs["y"] = adata.obs["y"] - adata.obs["y"].min()
adata.obs["z"] = adata.obs["z"] - 4.9
adata.obs[["x", "y", "z"]] = adata.obs[["x", "y", "z"]].round(2)
adata.obsm["spatial"] = adata.obs[["x", "y", "z"]].values
print(adata.obs["anno"].unique().tolist())

adata = adata[adata.obs["anno"] == "fat body", :].copy()
adata = sd.tl.om_kde(adata, threshold=0.4)
adata = sd.tl.om_EllipticEnvelope(adata, threshold=0.2)

pcd, volume = sd.tl.construct_three_d_mesh(
    adata=adata,
    groupby="anno",
    key_added="groups",
    mesh_style="volume",
    mesh_color="blue",
    pcd_cmap="rainbow",
    pcd_amap=0.8,
    pcd_voxelize=True,
    pcd_voxel_size=[0.5, 0.5, 0.5],
    cs_method="alpha_shape",
)
import pyvista as pv

new_mesh = volume.slice_orthogonal()
print(type(new_mesh))
"""
picked = sd.tl.three_d_split(mesh=pcd, key="groups")

import pyvista as pv

# convert these meshes back to surface meshes (PolyData)
separated_meshes = []
for selected_mesh in picked:
    separated_meshes.append(selected_mesh)
    pv.plot(selected_mesh)
pv.plot(separated_meshes)
"""
