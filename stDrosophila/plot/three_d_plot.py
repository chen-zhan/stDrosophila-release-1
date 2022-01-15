import warnings

import matplotlib as mpl
import numpy as np
import pandas as pd
import pyvista as pv
import PVGeo
import seaborn as sns

from anndata import AnnData
from pandas.core.frame import DataFrame
from typing import Union, Optional, Sequence, List


def clip_3d_coords(adata: AnnData,
                   coordsby: str = "spatial"
                   ):

    bucket_xyz = adata.obsm[coordsby].astype(float)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    grid = pv.PolyData(bucket_xyz)
    grid["index"] = adata.obs_names.to_numpy()
    # Clip mesh using a pyvista.PolyData surface mesh.
    surf = grid.delaunay_3d().extract_geometry()
    surf.subdivide(nsub=3, subfilter="loop", inplace=True)
    clipped_grid = grid.clip_surface(surf)
    clipped_adata = adata[clipped_grid["index"], :]

    clipped_points = pd.DataFrame()
    clipped_points[0] = list(map(tuple, clipped_grid.points))
    surf_points = list(map(tuple, surf.points.round(1)))
    clipped_points[1] = clipped_points[0].isin(surf_points)
    clipped_adata = clipped_adata[~clipped_points[1].values, :]

    return clipped_adata, surf


def three_d_color(series,
                  colormap: Union[str, list, dict] = None,
                  alphamap: Union[float, list, dict] = None,
                  mask_color: Optional[str] = None,
                  mask_alpha: Optional[float] = None
                  ):

    color_typies = series.unique().tolist()
    color_typies.sort()

    colordict = {}
    # set mask rgba
    if "mask" in color_typies:
        color_typies.remove("mask")
        rgb_color = mpl.colors.to_rgb(mask_color)
        colordict["mask"] = [rgb_color[0], rgb_color[1], rgb_color[2], mask_alpha]

    # set alpha
    if isinstance(alphamap, float) or isinstance(alphamap, int):
        alphamap = {t: alphamap for t in color_typies}
    elif isinstance(alphamap, list):
        alphamap = {t: alpha for t, alpha in zip(color_typies, alphamap)}
    # set rgb
    if isinstance(colormap, str):
        colormap = [mpl.colors.to_hex(i, keep_alpha=False) for i in
                    sns.color_palette(palette=colormap, n_colors=len(color_typies), as_cmap=False)]
    if isinstance(colormap, list):
        colormap = {t: color for t, color in zip(color_typies, colormap)}
    # set rgba
    for t in color_typies:
        rgb_color = mpl.colors.to_rgb(colormap[t])
        colordict[t] = [rgb_color[0], rgb_color[1], rgb_color[2], alphamap[t]]
    rgba = np.array([colordict[g] for g in series.tolist()])

    return rgba


def build_three_d_model(adata: AnnData,
                        coordsby: str = "spatial",
                        groupby: Optional[str] = "cluster",
                        group_show: Union[str, list] = "all",
                        group_cmap: Union[str, list, dict] = "viridis",
                        group_amap: Union[float, list, dict] = 1.0,
                        gene_show: Union[str, list] = "all",
                        gene_cmap: Union[str, list, dict] = "hot_r",
                        gene_amap: Union[float, list, dict] = 1.0,
                        mask_color: Optional[str] = "gainsboro",
                        mask_alpha: Optional[float] = 0.5,
                        smoothing: Optional[bool] = True,
                        voxelize: Optional[bool] = False,
                        voxel_size: Optional[list] = None,
                        unstructure: Optional[bool] = False
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

    # filter gene expression info
    genes_exp = _adata.X.sum(axis=1) if gene_show == "all" else _adata[:, gene_show].X.sum(axis=1)
    genes_exp = genes_exp.round(5).astype(float)
    genes_data = pd.DataFrame([groups.values.tolist(), genes_exp.tolist()]).stack().unstack(0)
    genes_data.columns = ["group", "genes_exp"]
    new_genes_exp = genes_data[["group", "genes_exp"]].apply(
        lambda x: "mask" if x["group"] is "mask" else x["genes_exp"], axis=1)

    # Create a point cloud(pyvista.PolyData) and its surface.
    bucket_xyz = adata.obsm[coordsby].astype(float)
    if isinstance(bucket_xyz, DataFrame) is False:
        bucket_xyz = pd.DataFrame(bucket_xyz)
    points = PVGeo.points_to_poly_data(bucket_xyz)
    surface = points.delaunay_3d().extract_geometry()

    if voxelize is True:
        # Create a voxelized volume(pyvista.UnstructuredGrid).
        voxelizer = PVGeo.filters.VoxelizePoints()
        voxel_size = [1, 1, 1] if voxel_size is None else voxel_size
        voxelizer.set_deltas(voxel_size[0], voxel_size[1], voxel_size[2])
        voxelizer.set_estimate_grid(False)
        mesh = voxelizer.apply(points)
    else:
        # Convert pyvista.PolyData format to pyvista.UnstructuredGrid format
        mesh = points.cast_to_unstructured_grid() if unstructure else points

    # Add some properties of the 3D model
    mesh["points_coords"] = bucket_xyz.values
    mesh["groups"] = groups.values
    mesh["genes_exp"] = genes_exp.values
    mesh["groups_rgba"] = three_d_color(series=groups, colormap=group_cmap, alphamap=group_amap,
                                        mask_color=mask_color, mask_alpha=mask_alpha)
    mesh["genes_rgba"] = three_d_color(series=new_genes_exp, colormap=gene_cmap, alphamap=gene_amap,
                                       mask_color=mask_color, mask_alpha=mask_alpha)

    return mesh, surface


def three_d_slicing(mesh,
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
        return mesh.slice(normal=axis, origin=center)
    else:
        # Create many slices of the input dataset along a specified axis.
        return mesh.slice_along_axis(n=n_slices, axis=axis, center=center)