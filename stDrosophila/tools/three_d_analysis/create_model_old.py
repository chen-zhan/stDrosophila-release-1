import matplotlib as mpl
import numpy as np
import open3d as o3d
import pandas as pd
import pyacvd
import pyvista as pv
import PVGeo
import seaborn as sns

from anndata import AnnData
from pandas.core.frame import DataFrame
from pyvista.core.pointset import PolyData, UnstructuredGrid
from typing import Optional, Tuple, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def create_points(
    adata: AnnData,
    coordsby: str = "spatial",
    ptype: Literal["polydata", "unstructured"] = "polydata",
    coodtype: type = np.float64,
) -> PolyData or UnstructuredGrid:
    """
    Create a point cloud based on 3D coordinate information.

    Args:
        adata: AnnData object.
        coordsby: The key from adata.obsm whose value will be used to reconstruct the 3D structure.
        ptype: The type of point cloud to output. The ptype must be `polydata` or `unstructured`.
        coodtype: Data type of 3D coordinate information.

    Returns:
        A point cloud.
    """

    bucket_xyz = adata.obsm[coordsby].astype(coodtype)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values

    if ptype == "polydata":
        return pv.PolyData(bucket_xyz)
    elif ptype == "unstructured":
        return pv.PolyData(bucket_xyz).cast_to_unstructured_grid()
    else:
        raise ValueError(
            "\n`ptype` value is wrong..The parameter `ptype` must be `polydata` or `unstructured`.\n"
        )


def create_surf(
    pcd: PolyData,
    cs_method: Literal[
        "basic", "slide", "alpha_shape", "ball_pivoting", "poisson"
    ] = "basic",
    cs_method_args: dict = None,
) -> PolyData:
    """
    Surface reconstruction from 3D point cloud.

    Args:
        pcd: A point cloud grid.
        cs_method: Create surface methods.
        cs_method_args: Parameters for various surface reconstruction methods.

    Returns:
        A surface mesh.
    """

    _cs_method_args = {
        "n_slide": 3,
        "al_alpha": 10,
        "ba_radii": [1, 1, 1, 1],
        "po_depth": 5,
        "po_threshold": 0.1,
    }
    if cs_method_args is not None:
        _cs_method_args.update(cs_method_args)

    if cs_method == "basic":
        return pcd.delaunay_3d().extract_surface()

    elif cs_method == "slide":
        n_slide = _cs_method_args["n_slide"]

        z_data = pd.Series(pcd.points[:, 2])
        layers = np.unique(z_data.tolist())
        n_layer_groups = len(layers) - n_slide + 1
        layer_groups = [layers[i : i + n_slide] for i in range(n_layer_groups)]

        points = np.empty(shape=[0, 3])
        for layer_group in layer_groups:
            lg_points = pcd.extract_points(z_data.isin(layer_group))

            lg_grid = lg_points.delaunay_3d().extract_surface()
            lg_grid.subdivide(nsub=2, subfilter="loop", inplace=True)

            points = np.concatenate((points, lg_grid.points), axis=0)

        return pv.PolyData(points).delaunay_3d().extract_surface()

    elif cs_method in ["alpha_shape", "ball_pivoting", "poisson"]:
        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(pcd.points)

        if cs_method == "alpha_shape":
            alpha = _cs_method_args["al_alpha"]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                _pcd, alpha
            )

        elif cs_method == "ball_pivoting":
            radii = _cs_method_args["ba_radii"]

            _pcd.normals = o3d.utility.Vector3dVector(
                np.zeros((1, 3))
            )  # invalidate existing normals
            _pcd.estimate_normals()

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                _pcd, o3d.utility.DoubleVector(radii)
            )

        else:
            depth, density_threshold = (
                _cs_method_args["po_depth"],
                _cs_method_args["po_threshold"],
            )

            _pcd.normals = o3d.utility.Vector3dVector(
                np.zeros((1, 3))
            )  # invalidate existing normals
            _pcd.estimate_normals()

            with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug
            ) as cm:
                (
                    mesh,
                    densities,
                ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    _pcd, depth=depth
                )
            mesh.remove_vertices_by_mask(
                np.asarray(densities) < np.quantile(densities, density_threshold)
            )

        _vertices = np.asarray(mesh.vertices)
        _faces = np.asarray(mesh.triangles)
        _faces = np.concatenate(
            (np.ones((_faces.shape[0], 1), dtype=np.int64) * 3, _faces), axis=1
        )

        return pv.PolyData(_vertices, _faces.ravel()).extract_surface()

    else:
        raise ValueError(
            "\n`method` value is wrong. Available `method` are: `basic` , `slide` ,`alpha_shape`, `ball_pivoting`, `poisson`."
        )


def smoothing_mesh(
    adata: AnnData,
    coordsby: str = "spatial",
    cs_method: Literal[
        "basic", "slide", "alpha_shape", "ball_pivoting", "poisson"
    ] = "basic",
    cs_method_args: dict = None,
    n_surf: int = 10000,
    coodtype: type = np.float64,
) -> Tuple[AnnData, PolyData]:
    """
    Takes a uniformly meshed surface using voronoi clustering and
    clip the original mesh using the reconstructed surface.

    Args:
        adata: AnnData object.
        coordsby: The key from adata.obsm whose value will be used to reconstruct the 3D structure.
        cs_method: Create surface methods.
        cs_method_args: Parameters for various surface reconstruction methods.
        n_surf: The number of faces obtained using voronoi clustering. The larger the number, the smoother the surface.
        coodtype: Data type of 3D coordinate information.

    Returns:
        clipped_adata: AnnData object that is clipped.
        uniform_surf: A uniformly meshed surface.
    """

    points = create_points(
        adata=adata, coordsby=coordsby, ptype="polydata", coodtype=coodtype
    )
    points.point_data["index"] = adata.obs_names.to_numpy()

    # takes a surface mesh and returns a uniformly meshed surface using voronoi clustering.
    surf = create_surf(pcd=points, cs_method=cs_method, cs_method_args=cs_method_args)
    surf.triangulate(inplace=True)
    surf.smooth(n_iter=100, inplace=True)
    surf.subdivide_adaptive(max_n_passes=2, inplace=True)
    clustered = pyacvd.Clustering(surf)
    # clustered.subdivide(3)
    clustered.cluster(n_surf)
    uniform_surf = clustered.create_mesh()

    # Clip the original mesh using the reconstructed surface.
    clipped_grid = points.clip_surface(uniform_surf)
    clipped_adata = adata[clipped_grid.point_data["index"], :]

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
    cs_method: Literal[
        "basic", "slide", "alpha_shape", "ball_pivoting", "poisson"
    ] = "basic",
    cs_method_args: dict = None,
    smoothing: bool = True,
    n_surf: int = 10000,
    voxelize: bool = True,
    voxel_size: Optional[list] = None,
    voxel_smooth: Optional[int] = 200,
    coodtype: type = np.float64,
    expdtype: type = np.float64,
) -> Tuple[UnstructuredGrid, UnstructuredGrid]:
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
        cs_method: Create surface methods.
        cs_method_args: Parameters for various surface reconstruction methods.
        smoothing: Smoothing the surface of the reconstructed 3D structure.
        n_surf: The number of faces obtained using voronoi clustering. The larger the n_surf, the smoother the surface. Only valid when smoothing is True.
        voxelize: Voxelize the reconstructed 3D structure.
        voxel_size: The size of the voxelized points. A list of three elements.
        voxel_smooth: The smoothness of the voxelized surface. Only valid when voxelize is True.
        coodtype: Data type of 3D coordinate information.
        expdtype: Data type of gene expression.
    Returns:
        mesh: Reconstructed 3D structure, which contains the following properties:
            groups: `mesh['groups']`, the mask and the groups used for display.
            genes_exp: `mesh['genes']`, the gene expression.
            groups_rgba: `mesh['groups_rgba']`, the rgba colors for plotting groups and mask.
            genes_rgba: `mesh['genes_rgba']`, the rgba colors for plotting genes and mask.
    """

    # takes a uniformly meshed surface and clip the original mesh using the reconstructed surface if smoothing is True.
    _adata, uniform_surf = (
        smoothing_mesh(
            adata=adata,
            coordsby=coordsby,
            cs_method=cs_method,
            cs_method_args=cs_method_args,
            n_surf=n_surf,
            coodtype=coodtype,
        )
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
            raise ValueError("\n`group_show` value is wrong.\n")

    # filter gene expression info
    genes_exp = (
        _adata.X.sum(axis=1)
        if gene_show == "all"
        else _adata[:, gene_show].X.sum(axis=1)
    )
    genes_exp = pd.DataFrame(genes_exp, index=groups.index).astype(expdtype)
    genes_data = pd.concat([groups, genes_exp], axis=1)
    genes_data.columns = ["groups", "genes_exp"]
    new_genes_exp = genes_data[["groups", "genes_exp"]].apply(
        lambda x: "mask" if x["groups"] is "mask" else round(x["genes_exp"], 2), axis=1
    )

    # Create a point cloud(Unstructured) and its surface.
    points = create_points(
        adata=_adata, coordsby=coordsby, ptype="unstructured", coodtype=coodtype
    )
    surface = (
        create_surf(pcd=points, cs_method=cs_method, cs_method_args=cs_method_args)
        if uniform_surf is None
        else uniform_surf
    )
    surface = surface.cast_to_unstructured_grid()

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
    ).astype(np.float64)

    points.cell_data["genes"] = (
        new_genes_exp.map(lambda x: 0 if x == "mask" else x).astype(expdtype).values
    )
    points.cell_data["genes_rgba"] = three_d_color(
        series=new_genes_exp,
        colormap=gene_cmap,
        alphamap=gene_amap,
        mask_color=mask_color,
        mask_alpha=mask_alpha,
    ).astype(np.float64)

    surface.cell_data["groups"] = np.array(["mask"] * surface.n_cells).astype(str)
    surface.cell_data["groups_rgba"] = np.array(
        [mpl.colors.to_rgba(surf_color, alpha=surf_alpha)] * surface.n_cells
    ).astype(np.float64)

    surface.cell_data["genes"] = np.array([0] * surface.n_cells).astype(expdtype)
    surface.cell_data["genes_rgba"] = np.array(
        [mpl.colors.to_rgba(surf_color, alpha=surf_alpha)] * surface.n_cells
    ).astype(np.float64)

    return surface, points


def merge_model(
    meshes: Union[
        List[PolyData or UnstructuredGrid], Tuple[PolyData or UnstructuredGrid]
    ],
) -> PolyData or UnstructuredGrid:
    """Merge all meshes in the `meshes` list. The format of all meshes must be the same."""

    merged_mesh = meshes[0]
    for mesh in meshes[1:]:
        merged_mesh.merge(mesh, inplace=True)

    return merged_mesh
