import re
import math
import warnings
import numpy as np
import pandas as pd
import anndata as ad
import pyvista as pv
import PVGeo
import matplotlib as mpl
import seaborn as sns
from anndata import AnnData
from typing import Union, Optional, Sequence, List


def clip_3d_coords(adata: AnnData,
                   coordsby: Optional[List] = None
                   ):

    if coordsby is None:
        coordsby = ["x", "y", "z"]
    points_data = adata.obs[coordsby]
    points_arr = points_data.values.astype(float)
    grid = pv.PolyData(points_arr)
    grid["index"] = adata.obs_names.tolist()
    # Clip mesh using a pyvista.PolyData surface mesh.
    surf = grid.delaunay_3d().extract_geometry()
    surf.subdivide(nsub=3, subfilter="loop", inplace=True)
    clipped_grid = grid.clip_surface(surf)
    clipped_adata = adata[adata.obs.index.isin(clipped_grid["index"].tolist()), :]

    return clipped_adata, surf


def group_color(groups,
                colormap: Union[str, list, dict] = "viridis",
                alphamap: Union[float, list, dict] = 1.0,
                mask_color: Optional[str] = "whitesmoke",
                mask_alpha: Optional[float] = 0.5
                ):

    color_groups = groups.unique().tolist()
    color_groups.sort()
    colordict = {}
    if "mask" in color_groups:
        color_groups.remove("mask")
        rgb_color = mpl.colors.to_rgb(mask_color)
        colordict["mask"] = [rgb_color[0], rgb_color[1], rgb_color[2], mask_alpha]

    # Set group color
    if isinstance(alphamap, float) or isinstance(alphamap, int):
        alphamap = {group: alphamap for group in color_groups}
    elif isinstance(alphamap, list):
        alphamap = {group: alpha for group, alpha in zip(color_groups, alphamap)}
    if isinstance(colormap, str):
        colormap = [mpl.colors.to_hex(i, keep_alpha=False) for i in
                    sns.color_palette(palette=colormap, n_colors=len(color_groups), as_cmap=False)]
    if isinstance(colormap, list):
        colormap = {group: color for group, color in zip(color_groups, colormap)}
    for group in color_groups:
        rgb_color = mpl.colors.to_rgb(colormap[group])
        colordict[group] = [rgb_color[0], rgb_color[1], rgb_color[2], alphamap[group]]

    return colordict


def build_3Dmodel(adata: AnnData,
                  coordsby: Optional[list] = None,
                  groupby: Optional[str] = "cluster",
                  group_show: Union[str, list] = "all",
                  gene_show: Union[str, list] = "all",
                  colormap: Union[str, list, dict] = "viridis",
                  alphamap: Union[float, list, dict] = 1.0,
                  mask_color: Optional[str] = "whitesmoke",
                  mask_alpha: Optional[float] = 0.5,
                  smoothing: Optional[bool] = True,
                  voxelize: Optional[bool] = True,
                  unstructure: Optional[bool] = True
                  ):
    # Clip mesh using a pyvista.PolyData surface mesh.
    if smoothing:
        _adata, _ = clip_3d_coords(adata=adata, coordsby=coordsby)
    else:
        _adata = adata

    # filter group info
    if isinstance(group_show, str) and group_show is "all":
        groups = _adata.obs[groupby]
    elif isinstance(group_show, str) and group_show is not "all":
        groups = _adata.obs[groupby].map(lambda x: str(x) if x == group_show else "mask")
    elif isinstance(group_show, list) or isinstance(group_show, tuple):
        groups = _adata.obs[groupby].map(lambda x: str(x) if x in group_show else "mask")
    else:
        raise ValueError("`group_show` value is wrong.")

    # Set group color(rgba)
    colordict = group_color(groups=groups, colormap=colormap, alphamap=alphamap,
                            mask_color=mask_color, mask_alpha=mask_alpha)

    # Create a point cloud(pyvista.PolyData) or a voxelized volume(pyvista.UnstructuredGrid).
    if coordsby is None:
        coordsby = ["x", "y", "z"]
    points_data = _adata.obs[coordsby]
    points_data = points_data.astype(float)
    points = PVGeo.points_to_poly_data(points_data)
    surface = points.delaunay_3d().extract_geometry()
    mesh = PVGeo.filters.VoxelizePoints().apply(points) if voxelize else points
    if unstructure and isinstance(mesh, pv.core.pointset.PolyData):
        mesh = mesh.cast_to_unstructured_grid()
    mesh["points_coords"] = points_data
    mesh["genes_exp"] = _adata.X.sum(axis=1) if gene_show == "all" else _adata[:, gene_show].X.sum(axis=1)
    mesh[groupby] = groups.values
    mesh[f"{groupby}_rgba"] = np.array([colordict[g] for g in groups.tolist()])

    return mesh, surface


def threeDslicing(mesh,
                  axis: Union[str, int] = "x",
                  n_slices: Union[str, int] = 10,
                  center: Optional[Sequence[float]] = None):

    if isinstance(mesh, pv.core.pointset.UnstructuredGrid) is False:
        warnings.warn("The model should be a pyvista.UnstructuredGrid (voxelized) object.")
        mesh = mesh.cast_to_unstructured_grid()

    if n_slices is "orthogonal":
        # Create three orthogonal slices through the dataset on the three cartesian planes.
        if center is None:
            return mesh.slice_orthogonal(x=None, y=None, z=None)
        else:
            return mesh.slice_orthogonal(x=center[0], y=center[1], z=center[2])
    elif n_slices == 1:
        # Slice a dataset by a plane at the specified origin and normal vector orientation.
        return mesh.slice(normal=axis, origin=center, contour=True)
    else:
        # Create many slices of the input dataset along a specified axis.
        return mesh.slice_along_axis(n=n_slices, axis=axis, center=center)


def easy_plot(surface=None, meshes=None, scalars=None, shape=(1, 1), cpos="iso", background_color="white", save=None):

    if isinstance(meshes, list) is False:
        meshes = [meshes]
    if isinstance(shape, str):
        n = re.split("[|/]", shape)
        subplot_index_list = [i for i in range(int(n[0]) + int(n[1]))]
    else:
        subplot_index_list = [[i, j] for i in range(shape[0]) for j in range(shape[1])]

    p = pv.Plotter(shape=shape)
    p.background_color = background_color
    for mesh, subplot_index in zip(meshes, subplot_index_list):
        if isinstance(subplot_index, int):
            p.subplot(subplot_index)
        elif isinstance(subplot_index, list):
            p.subplot(subplot_index[0], subplot_index[1])
        if surface is not None:
            p.add_mesh(surface, color="whitesmoke", opacity=0.2)
        p.add_mesh(mesh, scalars=scalars, rgba=True)
    p.camera_position = cpos
    p.show(screenshot=save)







