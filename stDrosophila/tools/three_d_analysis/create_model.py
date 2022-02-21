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


def mesh_type(
    mesh: Union[PolyData, UnstructuredGrid],
    mtype: Literal["polydata", "unstructured"] = "polydata",
) -> PolyData or UnstructuredGrid:
    """Get a new representation of this mesh as a new type."""
    if mtype == "polydata":
        return (
            mesh if isinstance(mesh, PolyData) else pv.PolyData(mesh.points, mesh.cells)
        )

    elif mtype == "unstructured":
        return mesh.cast_to_unstructured_grid() if isinstance(mesh, PolyData) else mesh

    else:
        raise ValueError(
            "\n`mtype` value is wrong."
            "\nAvailable `mtype` are: `polydata` and `unstructured`.\n"
        )


def construct_pcd(
    adata: AnnData,
    coordsby: str = "spatial",
    mtype: Literal["polydata", "unstructured"] = "polydata",
    coodtype: type = np.float64,
) -> PolyData or UnstructuredGrid:
    """
    Construct a point cloud model based on 3D coordinate information.

    Args:
        adata: AnnData object.
        coordsby: The key from adata.obsm whose value will be used to reconstruct the 3D structure.
        mtype: The type of the point cloud. Available `mtype` are: `polydata` and `unstructured`.
        coodtype: Data type of 3D coordinate information.

    Returns:
        A point cloud.
    """

    bucket_xyz = adata.obsm[coordsby].astype(coodtype)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    pcd = pv.PolyData(bucket_xyz)

    return mesh_type(mesh=pcd, mtype=mtype)


def voxelize_pcd(
    pcd: PolyData,
    voxel_size: Optional[list] = None,
):
    """
    Voxelize the point cloud.

    Args:
        pcd: A point cloud.
        voxel_size: The size of the voxelized points. A list of three elements.
    Returns:
        A voxelized point cloud.
    """

    voxel_size = [1, 1, 1] if voxel_size is None else voxel_size

    voxelizer = PVGeo.filters.VoxelizePoints()
    voxelizer.set_deltas(voxel_size[0], voxel_size[1], voxel_size[2])
    voxelizer.set_estimate_grid(False)

    return voxelizer.apply(pcd)


def construct_surface(
    pcd: PolyData,
    cs_method: Literal[
        "basic", "slide", "alpha_shape", "ball_pivoting", "poisson"
    ] = "basic",
    cs_method_args: dict = None,
    surface_smoothness: int = 100,
    n_surf: int = 10000,
    mtype: Literal["polydata", "unstructured"] = "polydata",
) -> PolyData or UnstructuredGrid:
    """
    Surface mesh reconstruction based on 3D point cloud model.

    Args:
        pcd: A point cloud.
        cs_method: Create surface methods.
        cs_method_args: Parameters for various surface reconstruction methods.
        surface_smoothness: Adjust surface point coordinates using Laplacian smoothing.
                            If smoothness==0, do not smooth the reconstructed surface.
        n_surf: The number of faces obtained using voronoi clustering. The larger the number, the smoother the surface.
        mtype: The type of the reconstructed surface. Available `mtype` are: `polydata` and `unstructured`.

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

    # Reconstruct surface mesh.
    if cs_method == "basic":
        surf = pcd.delaunay_3d().extract_surface()

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

        surf = pv.PolyData(points).delaunay_3d().extract_surface()

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
        surf = pv.PolyData(_vertices, _faces.ravel()).extract_surface()

    else:
        raise ValueError(
            "\n`cs_method` value is wrong."
            "\nAvailable `cs_method` are: `basic` , `slide` ,`alpha_shape`, `ball_pivoting`, `poisson`.\n"
        )

    # Get an all triangle mesh.
    surf.triangulate(inplace=True)

    # Smooth the reconstructed surface.
    if surface_smoothness != 0:
        surf.smooth(n_iter=surface_smoothness, inplace=True)
        surf.subdivide_adaptive(max_n_passes=3, inplace=True)

    # Get a uniformly meshed surface using voronoi clustering.
    clustered = pyacvd.Clustering(surf)
    clustered.cluster(n_surf)
    uniform_surf = clustered.create_mesh()

    return mesh_type(mesh=uniform_surf, mtype=mtype)


def construct_volume(
    mesh: Union[PolyData, UnstructuredGrid],
    volume_smoothness: Optional[int] = 200,
):
    """Construct a volumetric mesh based on surface mesh.

    Args:
        mesh: A surface mesh.
        volume_smoothness: The smoothness of the volumetric mesh.

    Returns:
        A volumetric mesh.
    """

    density = mesh.length / volume_smoothness
    volume = pv.voxelize(mesh, density=density, check_surface=False)

    return volume


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

    # Set mask rgba.
    if "mask" in color_types:
        color_types.remove("mask")
        colordict["mask"] = mpl.colors.to_rgba(mask_color, alpha=mask_alpha)
    color_types.sort()

    # Set alpha.
    if isinstance(alphamap, float) or isinstance(alphamap, int):
        alphamap = {t: alphamap for t in color_types}
    elif isinstance(alphamap, list):
        alphamap = {t: alpha for t, alpha in zip(color_types, alphamap)}

    # Set rgb.
    if isinstance(colormap, str):
        colormap = [
            mpl.colors.to_hex(i, keep_alpha=False)
            for i in sns.color_palette(
                palette=colormap, n_colors=len(color_types), as_cmap=False
            )
        ]
    if isinstance(colormap, list):
        colormap = {t: color for t, color in zip(color_types, colormap)}

    # Set rgba.
    for t in color_types:
        colordict[t] = mpl.colors.to_rgba(colormap[t], alpha=alphamap[t])
    rgba = np.array([colordict[g] for g in series.tolist()])

    return rgba


def construct_three_d_mesh(
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
    cs_method: Literal[
        "basic", "slide", "alpha_shape", "ball_pivoting", "poisson"
    ] = "basic",
    cs_method_args: dict = None,
    surf_smoothness: int = 100,
    n_surf: int = 10000,
    vol_color: str = "gainsboro",
    vol_alpha: float = 0.5,
    vol_smoothness: Optional[int] = 200,
    pcd_voxelize: bool = True,
    pcd_voxel_size: Optional[list] = None,
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
        vol_color: Color to use for plotting surface. The default mask_color is `'gainsboro'`.
        vol_alpha: The opacity of the color to use for plotting surface. The default mask_alpha is `0.5`.
        vol_smoothness: The smoothness of the volumetric mesh.
        cs_method: Create surface methods.
        cs_method_args: Parameters for various surface reconstruction methods.
        surf_smoothness: Adjust surface point coordinates using Laplacian smoothing.
                         If surf_smoothness==0, do not smooth the reconstructed surface.
        n_surf: The number of faces obtained using voronoi clustering. The larger the n_surf, the smoother the surface.
                Only valid when smoothing is True.
        pcd_voxelize: Voxelize the reconstructed 3D structure.
        pcd_voxel_size: The size of the voxelized points. A list of three elements.
        coodtype: Data type of 3D coordinate information.
        expdtype: Data type of gene expression.
    Returns:
        new_pcd: Reconstructed 3D point cloud, which contains the following properties:
            groups: `new_pcd['groups']`, the mask and the groups used for display.
            genes_exp: `new_pcd['genes']`, the gene expression.
            groups_rgba: `new_pcd['groups_rgba']`, the rgba colors for plotting groups and mask.
            genes_rgba: `new_pcd['genes_rgba']`, the rgba colors for plotting genes and mask.
        volume: Reconstructed volumetric mesh, which contains the following properties:
            groups: `volume['groups']`, the mask and the groups used for display.
            genes_exp: `volume['genes']`, the gene expression.
            groups_rgba: `volume['groups_rgba']`, the rgba colors for plotting groups and mask.
            genes_rgba: `volume['genes_rgba']`, the rgba colors for plotting genes and mask.
    """

    # Reconstruct a point cloud and a volumetric mesh.
    raw_pcd = construct_pcd(
        adata=adata, coordsby=coordsby, mtype="polydata", coodtype=coodtype
    )
    raw_pcd.point_data["index"] = adata.obs_names.to_numpy()
    surface = construct_surface(
        pcd=raw_pcd,
        cs_method=cs_method,
        cs_method_args=cs_method_args,
        surface_smoothness=surf_smoothness,
        n_surf=n_surf,
        mtype="polydata",
    )
    volume = construct_volume(mesh=surface, volume_smoothness=vol_smoothness)

    # Clip the original pcd using the reconstructed surface and reconstruct new point cloud.
    clipped_pcd = raw_pcd.clip_surface(surface)
    clipped_adata = adata[clipped_pcd.point_data["index"], :]
    new_pcd = construct_pcd(
        adata=clipped_adata, coordsby=coordsby, mtype="polydata", coodtype=coodtype
    )
    if pcd_voxelize:
        new_pcd = voxelize_pcd(pcd=new_pcd, voxel_size=pcd_voxel_size)

    # Filter group info.
    if groupby is None:
        n_points = clipped_adata.obs.shape[0]
        groups = pd.Series(
            ["same"] * n_points, index=clipped_adata.obs.index, dtype=str
        )
    else:
        if isinstance(group_show, str) and group_show is "all":
            groups = clipped_adata.obs[groupby]
        elif isinstance(group_show, str) and group_show is not "all":
            groups = clipped_adata.obs[groupby].map(
                lambda x: str(x) if x == group_show else "mask"
            )
        elif isinstance(group_show, list) or isinstance(group_show, tuple):
            groups = clipped_adata.obs[groupby].map(
                lambda x: str(x) if x in group_show else "mask"
            )
        else:
            raise ValueError("\n`group_show` value is wrong.\n")

    # Filter gene expression info.
    genes_exp = (
        clipped_adata.X.sum(axis=1)
        if gene_show == "all"
        else clipped_adata[:, gene_show].X.sum(axis=1)
    )
    genes_exp = pd.DataFrame(genes_exp, index=groups.index).astype(expdtype)
    genes_data = pd.concat([groups, genes_exp], axis=1)
    genes_data.columns = ["groups", "genes_exp"]
    new_genes_exp = genes_data[["groups", "genes_exp"]].apply(
        lambda x: "mask" if x["groups"] is "mask" else round(x["genes_exp"], 2), axis=1
    )

    # Add some properties of the 3D model
    new_pcd.cell_data["groups"] = groups.astype(str).values
    new_pcd.cell_data["groups_rgba"] = three_d_color(
        series=groups,
        colormap=group_cmap,
        alphamap=group_amap,
        mask_color=mask_color,
        mask_alpha=mask_alpha,
    ).astype(np.float64)

    new_pcd.cell_data["genes"] = (
        new_genes_exp.map(lambda x: 0 if x == "mask" else x).astype(expdtype).values
    )
    new_pcd.cell_data["genes_rgba"] = three_d_color(
        series=new_genes_exp,
        colormap=gene_cmap,
        alphamap=gene_amap,
        mask_color=mask_color,
        mask_alpha=mask_alpha,
    ).astype(np.float64)

    volume.cell_data["groups"] = np.array(["mask"] * volume.n_cells).astype(str)
    volume.cell_data["groups_rgba"] = np.array(
        [mpl.colors.to_rgba(vol_color, alpha=vol_alpha)] * volume.n_cells
    ).astype(np.float64)

    volume.cell_data["genes"] = np.array([0] * volume.n_cells).astype(expdtype)
    volume.cell_data["genes_rgba"] = np.array(
        [mpl.colors.to_rgba(vol_color, alpha=vol_alpha)] * volume.n_cells
    ).astype(np.float64)

    return new_pcd, volume


def merge_mesh(
    meshes: Union[
        List[PolyData or UnstructuredGrid], Tuple[PolyData or UnstructuredGrid]
    ],
) -> PolyData or UnstructuredGrid:
    """Merge all meshes in the `meshes` list. The format of all meshes must be the same."""

    merged_mesh = meshes[0]
    for mesh in meshes[1:]:
        merged_mesh.merge(mesh, inplace=True)

    return merged_mesh
