import anndata as ad
import numpy as np
import stDrosophila as sd
from pyvista import MultiBlock
import vtk

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

mesh = sd.tl.merge_mesh([pcd, volume])
raw_mesh = mesh.copy()

import pyvista as pv
import pandas as pd

p = pv.Plotter()
p.add_mesh(mesh, scalars=f"groups_rgba", rgba=True)
picked_meshes, legend = [], []
invert_meshes = []


def split_mesh(original_mesh):
    """Adds a new mesh to the plotter each time cells are picked, and
    removes them from the original mesh"""

    # if nothing selected
    if not original_mesh.n_cells:
        return

    # remove the picked cells from main grid
    ghost_cells = np.zeros(mesh.n_cells, np.uint8)
    ghost_cells[original_mesh["orig_extract_id"]] = 1
    mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = ghost_cells
    mesh.RemoveGhostCells()

    # add the selected mesh this to the main plotter
    color = np.random.random(3)
    legend.append(["picked mesh %d" % len(picked_meshes), color])
    p.add_mesh(original_mesh, color=color)
    p.add_legend(legend)

    # track the picked meshes and label them
    original_mesh["picked_index"] = np.ones(original_mesh.n_points) * len(picked_meshes)
    picked_meshes.append(original_mesh)
    invert_meshes.append(mesh)


# enable cell picking with our custom callback
p.enable_cell_picking(
    mesh=mesh,
    callback=split_mesh,
    show=False,
    font_size=12,
    show_message="Press `r` to enable retangle based selection. Press `r` again to turn it off. ",
)
p.show()
print(invert_meshes)
invert_mesh = invert_meshes[0]
invert_mesh.plot()
