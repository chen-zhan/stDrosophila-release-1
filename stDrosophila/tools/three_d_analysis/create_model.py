
import matplotlib as mpl
import numpy as np
import pandas as pd
import pyacvd
import pyvista as pv
import PVGeo
import seaborn as sns

from anndata import AnnData
from pandas.core.frame import DataFrame
from pyvista.core.pointset import PolyData, UnstructuredGrid
from typing import Optional, Tuple, Union


def smoothing_mesh(
    adata: AnnData, coordsby: str = "spatial", n_surf: int = 10000
) -> Tuple[AnnData, PolyData]:
    """
    Takes a uniformly meshed surface using voronoi clustering and
    clip the original mesh using the reconstructed surface.

    Args:
        adata: AnnData object.
        coordsby: The key from adata.obsm whose value will be used to reconstruct the 3D structure.
        n_surf: The number of faces obtained using voronoi clustering. The larger the number, the smoother the surface.

    Returns:
        clipped_adata: AnnData object that is clipped.
        uniform_surf: A uniformly meshed surface.
    """

    float_type = np.float64

    bucket_xyz = adata.obsm[coordsby].astype(float_type)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    grid = pv.PolyData(bucket_xyz)
    grid["index"] = adata.obs_names.to_numpy()

    # takes a surface mesh and returns a uniformly meshed surface using voronoi clustering.
    surf = grid.delaunay_3d().extract_geometry()
    #surf.smooth(n_iter=1000)
    surf.subdivide(nsub=3, subfilter="loop", inplace=True)
    clustered = pyacvd.Clustering(surf)
    # clustered.subdivide(3)
    clustered.cluster(n_surf)
    uniform_surf = clustered.create_mesh()

    # Clip the original mesh using the reconstructed surface.
    clipped_grid = grid.clip_surface(uniform_surf)
    clipped_adata = adata[clipped_grid["index"], :]

    clipped_points = pd.DataFrame()
    clipped_points[0] = list(map(tuple, clipped_grid.points.round(5)))
    surf_points = list(map(tuple, uniform_surf.points.round(5)))
    clipped_points[1] = clipped_points[0].isin(surf_points)
    clipped_adata = clipped_adata[~clipped_points[1].values, :]

    return clipped_adata, uniform_surf


def three_d_color(
    series,
    colormap: Union[str, list, dict] = None,
    alphamap: Union[float, list, dict] = None,
    mask_color: Optional[str] = None,
    mask_alpha: Optional[float] = None,
) -> np.ndarray:
    """
    Set the color of groups or gene expression.
    Args:
        series: Pandas sereis (e.g. cell groups or gene names).
        colormap: Colors to use for plotting data.
        alphamap: The opacity of the color to use for plotting data.
        mask_color: Colors to use for plotting mask information.
        mask_alpha: The opacity of the color to use for plotting mask information.
    Returns:
        rgba: The rgba values mapped to groups or gene expression.
    """

    color_types = series.unique().tolist()
    colordict = {}

    # set mask rgba
    if "mask" in color_types:
        color_types.remove("mask")
        colordict["mask"] = mpl.colors.to_rgba(mask_color, alpha=mask_alpha)
    color_types.sort()

    # set alpha
    if isinstance(alphamap, float) or isinstance(alphamap, int):
        alphamap = {t: alphamap for t in color_types}
    elif isinstance(alphamap, list):
        alphamap = {t: alpha for t, alpha in zip(color_types, alphamap)}

    # set rgb
    if isinstance(colormap, str):
        colormap = [
            mpl.colors.to_hex(i, keep_alpha=False)
            for i in sns.color_palette(
                palette=colormap, n_colors=len(color_types), as_cmap=False
            )
        ]
    if isinstance(colormap, list):
        colormap = {t: color for t, color in zip(color_types, colormap)}

    # set rgba
    for t in color_types:
        colordict[t] = mpl.colors.to_rgba(colormap[t], alpha=alphamap[t])
    rgba = np.array([colordict[g] for g in series.tolist()])

    return rgba


def build_three_d_model(
    adata: AnnData,
    coordsby: str = "spatial",
    groupby: Optional[str] = None,
    group_show: Union[str, list] = "all",
    group_cmap: Union[str, list, dict] = "rainbow",
    group_amap: Union[float, list, dict] = 1.0,
    gene_show: Union[str, list] = "all",
    gene_cmap: str = "hot_r",
    gene_amap: float = 1.0,
    mask_color: str = "gainsboro",
    mask_alpha: float = 0,
    surf_color: str = "gainsboro",
    surf_alpha: float = 0.5,
    smoothing: bool = True,
    n_surf: int = 10000,
    voxelize: bool = True,
    voxel_size: Optional[list] = None,
    voxel_smooth: Optional[int] = 200,
) -> UnstructuredGrid:
    """
    Reconstruct a voxelized 3D model.
    Args:
        adata: AnnData object.
        coordsby: The key from adata.obsm whose value will be used to reconstruct the 3D structure.
        groupby: The key of the observations grouping to consider.
        group_show: Subset of groups used for display, e.g. [`'g1'`, `'g2'`, `'g3'`]. The default group_show is `'all'`, for all groups.
        group_cmap: Colors to use for plotting groups. The default group_cmap is `'rainbow'`.
        group_amap: The opacity of the colors to use for plotting groups. The default group_amap is `1.0`.
        gene_show: Subset of genes used for display, e.g. [`'g1'`, `'g2'`, `'g3'`]. The default gene_show is `'all'`, for all groups.
        gene_cmap: Colors to use for plotting genes. The default gene_cmap is `'hot_r'`.
        gene_amap: The opacity of the colors to use for plotting genes. The default gene_amap is `1.0`.
        mask_color: Color to use for plotting mask. The default mask_color is `'gainsboro'`.
        mask_alpha: The opacity of the color to use for plotting mask. The default mask_alpha is `0.0`.
        surf_color: Color to use for plotting surface. The default mask_color is `'gainsboro'`.
        surf_alpha: The opacity of the color to use for plotting surface. The default mask_alpha is `0.5`.
        smoothing: Smoothing the surface of the reconstructed 3D structure.
        n_surf: The number of faces obtained using voronoi clustering. The larger the n_surf, the smoother the surface. Only valid when smoothing is True.
        voxelize: Voxelize the reconstructed 3D structure.
        voxel_size: The size of the voxelized points. A list of three elements.
        voxel_smooth: The smoothness of the voxelized surface. Only valid when voxelize is True.
    Returns:
        mesh: Reconstructed 3D structure, which contains the following properties:
            groups: `mesh['groups']`, the mask and the groups used for display.
            genes_exp: `mesh['genes']`, the gene expression.
            groups_rgba: `mesh['groups_rgba']`, the rgba colors for plotting groups and mask.
            genes_rgba: `mesh['genes_rgba']`, the rgba colors for plotting genes and mask.
    """

    float_type = np.float64

    # takes a uniformly meshed surface and clip the original mesh using the reconstructed surface if smoothing is True.
    _adata, uniform_surf = (
        smoothing_mesh(adata=adata, coordsby=coordsby, n_surf=n_surf)
        if smoothing
        else (adata, None)
    )

    # filter group info
    if groupby is None:
        n_points = _adata.obs.shape[0]
        groups = pd.Series(["same"] * n_points, index=_adata.obs.index, dtype=str)
    else:
        if isinstance(group_show, str) and group_show is "all":
            groups = _adata.obs[groupby]
        elif isinstance(group_show, str) and group_show is not "all":
            groups = _adata.obs[groupby].map(
                lambda x: str(x) if x == group_show else "mask"
            )
        elif isinstance(group_show, list) or isinstance(group_show, tuple):
            groups = _adata.obs[groupby].map(
                lambda x: str(x) if x in group_show else "mask"
            )
        else:
            raise ValueError("`group_show` value is wrong.")

    # filter gene expression info
    genes_exp = (
        _adata.X.sum(axis=1)
        if gene_show == "all"
        else _adata[:, gene_show].X.sum(axis=1)
    )
    genes_exp = pd.DataFrame(genes_exp, index=groups.index, dtype=float_type)
    genes_data = pd.concat([groups, genes_exp], axis=1)
    genes_data.columns = ["groups", "genes_exp"]
    new_genes_exp = genes_data[["groups", "genes_exp"]].apply(
        lambda x: "mask" if x["groups"] is "mask" else round(x["genes_exp"], 2), axis=1
    )

    # Create a point cloud(Unstructured) and its surface.
    bucket_xyz = _adata.obsm[coordsby].astype(float_type)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    points = pv.PolyData(bucket_xyz).cast_to_unstructured_grid()
    surface = (
        points.delaunay_3d().extract_geometry()
        if uniform_surf is None
        else uniform_surf
    )

    # Voxelize the cloud and the surface
    if voxelize:
        voxelizer = PVGeo.filters.VoxelizePoints()
        voxel_size = [1, 1, 1] if voxel_size is None else voxel_size
        voxelizer.set_deltas(voxel_size[0], voxel_size[1], voxel_size[2])
        voxelizer.set_estimate_grid(False)
        points = voxelizer.apply(points)

        density = surface.length / voxel_smooth
        surface = pv.voxelize(surface, density=density, check_surface=False)

    # Add some properties of the 3D model
    points.cell_data["groups"] = groups.astype(str).values
    points.cell_data["groups_rgba"] = three_d_color(
        series=groups,
        colormap=group_cmap,
        alphamap=group_amap,
        mask_color=mask_color,
        mask_alpha=mask_alpha,
    ).astype(float_type)

    points.cell_data["genes"] = (
        new_genes_exp.map(lambda x: 0 if x == "mask" else x).astype(float_type).values
    )
    points.cell_data["genes_rgba"] = three_d_color(
        series=new_genes_exp,
        colormap=gene_cmap,
        alphamap=gene_amap,
        mask_color=mask_color,
        mask_alpha=mask_alpha,
    ).astype(float_type)

    surface.cell_data["groups"] = np.array(["mask"] * surface.n_cells).astype(str)
    surface.cell_data["groups_rgba"] = np.array(
        [mpl.colors.to_rgba(surf_color, alpha=surf_alpha)] * surface.n_cells
    ).astype(float_type)

    surface.cell_data["genes"] = np.array([0] * surface.n_cells).astype(float_type)
    surface.cell_data["genes_rgba"] = np.array(
        [mpl.colors.to_rgba(surf_color, alpha=surf_alpha)] * surface.n_cells
    ).astype(float_type)

    # Merge points and surface into a single mesh.
    mesh = surface.merge(points)

    return mesh
