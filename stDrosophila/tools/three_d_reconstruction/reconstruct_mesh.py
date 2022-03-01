import matplotlib as mpl
import numpy as np
import open3d as o3d
import pandas as pd
import pyacvd
import pymeshfix as mf
import pyvista as pv
import PVGeo

from anndata import AnnData
from pandas.core.frame import DataFrame
from pyvista import PolyData, UnstructuredGrid, MultiBlock, DataSet
from typing import Optional, Tuple, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def mesh_type(
    mesh: Union[PolyData, UnstructuredGrid],
    mtype: Literal["polydata", "unstructuredgrid"] = "polydata",
) -> PolyData or UnstructuredGrid:
    """Get a new representation of this mesh as a new type."""
    if mtype == "polydata":
        return mesh if isinstance(mesh, PolyData) else pv.PolyData(mesh.points, mesh.cells)
    elif mtype == "unstructured":
        return mesh.cast_to_unstructured_grid() if isinstance(mesh, PolyData) else mesh
    else:
        raise ValueError("\n`mtype` value is wrong." 
                         "\nAvailable `mtype` are: `'polydata'` and `'unstructuredgrid'`.")


def construct_pcd(
    adata: AnnData,
    coordsby: str = "spatial",
    groupby: Optional[str] = None,
    key_added: str = "groups",
    mask: Union[str, int, float, list] = None,
    colormap: Union[str, list, dict] = "rainbow",
    alphamap: Union[float, list, dict] = 1.0,
    coodtype: type = np.float64,
) -> PolyData or UnstructuredGrid:
    """
    Construct a point cloud model based on 3D coordinate information.

    Args:
        adata: AnnData object.
        coordsby: The key from adata.obsm whose value will be used to reconstruct the 3D structure.
        groupby: The key of the observations grouping to consider.
        key_added: The key under which to add the labels.
        mask: The part that you don't want to be displayed.
        colormap: Colors to use for plotting pcd. The default pcd_cmap is `'rainbow'`.
        alphamap: The opacity of the colors to use for plotting pcd. The default pcd_amap is `1.0`.
        coodtype: Data type of 3D coordinate information.

    Returns:
        pcd: A point cloud, which contains the following properties:
            `pcd.point_data["obs_index"]`, the obs_index of each coordinate in the original adata.
            `pcd.point_data[key_added]`, the `groupby` array;
            `pcd.point_data[f'{key_added}_rgba']`, the rgba colors of the labels.
    """

    # create an initial pcd.
    bucket_xyz = adata.obsm[coordsby].astype(coodtype)
    if isinstance(bucket_xyz, DataFrame):
        bucket_xyz = bucket_xyz.values
    pcd = pv.PolyData(bucket_xyz)

    # The obs_index of each coordinate in the original adata.
    pcd.point_data["obs_index"] = adata.obs_names.to_numpy()

    # The`groupby` array in original adata.obs or adata.X
    mask_list = mask if isinstance(mask, list) else [mask]

    if groupby in adata.obs.columns:
        groups = adata.obs[groupby].map(lambda x: "mask" if x in mask_list else x).values
    elif groupby in adata.var.index:
        groups = adata[:, groupby].X.flatten()
    elif groupby is None:
        groups = np.array(["same"] * adata.obs.shape[0])
    else:
        raise ValueError(
            "\n`groupby` value is wrong." 
            "\n`groupby` should be one of adata.obs.columns, or one of adata.var.index"
        )

    pcd = add_mesh_labels(
        mesh=pcd,
        labels=groups,
        key_added=key_added,
        colormap=colormap,
        alphamap=alphamap
    )

    return pcd


def voxelize_pcd(
    pcd: PolyData,
    voxel_size: Optional[list] = None,
) -> UnstructuredGrid:
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
    key_added: str = "groups",
    color: Optional[str] = "gainsboro",
    alpha: Optional[float] = 0.8,
    cs_method: Literal["basic", "slide", "alpha_shape", "ball_pivoting", "poisson"] = "basic",
    cs_method_args: dict = None,
    surface_smoothness: int = 100,
    n_surf: int = 10000,
) -> Tuple[PolyData, PolyData]:
    """
    Surface mesh reconstruction based on 3D point cloud model.

    Args:
        pcd: A point cloud.
        key_added: The key under which to add the labels.
        color: Color to use for plotting mesh. The default mesh_color is `'gainsboro'`.
        alpha: The opacity of the color to use for plotting mesh. The default mesh_color is `0.8`.
        cs_method: The methods of creating a surface mesh. Available `cs_method` are:
                * `'basic'`
                * `'slide'`
                * `'alpha_shape'`
                * `'ball_pivoting'`
                * `'poisson'`
        cs_method_args: Parameters for various surface reconstruction methods. Available `cs_method_args` are:
                * `'slide'` method: {"n_slide": 3}
                * `'alpha_shape'` method: {"al_alpha": 10}
                * `'ball_pivoting'` method: {"ba_radii": [1, 1, 1, 1]}
                * `'poisson'` method: {"po_depth": 5, "po_threshold": 0.1}
        surface_smoothness: Adjust surface point coordinates using Laplacian smoothing.
                            If smoothness==0, do not smooth the reconstructed surface.
        n_surf: The number of faces obtained using voronoi clustering. The larger the number, the smoother the surface.
    Returns:
        uniform_surf: A reconstructed surface mesh, which contains the following properties:
            `surf.point_data[key_added]`, the "surface" array;
            `surf.point_data[f'{key_added}_rgba']`, the rgba colors of the labels.
        clipped_pcd: A point cloud, which contains the following properties:
            `clipped_pcd.point_data["obs_index"]`, the obs_index of each coordinate in the original adata.
            `clipped_pcd.point_data[key_added]`, the `groupby` array;
            `clipped_pcd.point_data[f'{key_added}_rgba']`, the rgba colors of the labels.
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
        layer_groups = [layers[i: i + n_slide] for i in range(n_layer_groups)]

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
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(_pcd, alpha)

        elif cs_method == "ball_pivoting":
            radii = _cs_method_args["ba_radii"]

            _pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
            _pcd.estimate_normals()

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                _pcd, o3d.utility.DoubleVector(radii)
            )

        else:
            depth, density_threshold = (
                _cs_method_args["po_depth"],
                _cs_method_args["po_threshold"],
            )

            _pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
            _pcd.estimate_normals()

            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                (
                    mesh,
                    densities,
                ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(_pcd, depth=depth)
            mesh.remove_vertices_by_mask(np.asarray(densities) < np.quantile(densities, density_threshold))

        _vertices = np.asarray(mesh.vertices)
        _faces = np.asarray(mesh.triangles)
        _faces = np.concatenate((np.ones((_faces.shape[0], 1), dtype=np.int64) * 3, _faces), axis=1)
        surf = pv.PolyData(_vertices, _faces.ravel()).extract_surface()

    else:
        raise ValueError(
            "\n`cs_method` value is wrong."
            "\nAvailable `cs_method` are: `'basic'` , `'slide'` ,`'alpha_shape'`, `'ball_pivoting'`, `'poisson'`."
        )

    # Get an all triangle mesh.
    surf.triangulate(inplace=True)

    # Repair the surface mesh where it was extracted and subtle holes along complex parts of the mesh
    meshfix = mf.MeshFix(surf)
    meshfix.repair(verbose=False)
    surf = meshfix.mesh

    # Smooth the reconstructed surface.
    if surface_smoothness != 0:
        surf.smooth(n_iter=surface_smoothness, inplace=True)
        surf.subdivide_adaptive(max_n_passes=3, inplace=True)

    # Get a uniformly meshed surface using voronoi clustering.
    clustered = pyacvd.Clustering(surf)
    clustered.cluster(n_surf)
    uniform_surf = clustered.create_mesh()

    # Add labels and the colormap of the surface mesh
    labels = np.array(["surface"] * uniform_surf.n_points).astype(str)
    uniform_surf = add_mesh_labels(
        mesh=uniform_surf,
        labels=labels,
        key_added=key_added,
        colormap=color,
        alphamap=alpha
    )

    # Clip the original pcd using the reconstructed surface and reconstruct new point cloud.
    clip_invert = True if cs_method in ["basic", "slide"] else False
    clipped_pcd = pcd.clip_surface(uniform_surf, invert=clip_invert)

    return uniform_surf, clipped_pcd


def construct_volume(
    mesh: Union[PolyData, UnstructuredGrid],
    key_added: str = "groups",
    color: Optional[str] = "gainsboro",
    alpha: Optional[float] = 0.8,
    volume_smoothness: Optional[int] = 200,
) -> UnstructuredGrid:
    """
    Construct a volumetric mesh based on surface mesh.

    Args:
        mesh: A surface mesh.
        key_added: The key under which to add the labels.
        color: Color to use for plotting mesh. The default mesh_color is `'gainsboro'`.
        alpha: The opacity of the color to use for plotting mesh. The default mesh_color is `0.8`.
        volume_smoothness: The smoothness of the volumetric mesh.

    Returns:
        volume: A reconstructed volumetric mesh, which contains the following properties:
            `volume.point_data[key_added]`, the "volume" array;
            `volume.point_data[f'{key_added}_rgba']`,  the rgba colors of the labels.

    """

    density = mesh.length / volume_smoothness
    volume = pv.voxelize(mesh, density=density, check_surface=False)

    # Add labels and the colormap of the volumetric mesh
    labels = np.array(["volume"] * volume.n_points).astype(str)
    volume = add_mesh_labels(
        mesh=volume,
        labels=labels,
        key_added=key_added,
        colormap=color,
        alphamap=alpha
    )

    return volume


def add_mesh_labels(
    mesh: Union[PolyData, UnstructuredGrid],
    labels: np.ndarray,
    key_added: str = "groups",
    colormap: Union[str, list] = None,
    alphamap: Union[float, list] = None,
    mask_color: Optional[str] = "gainsboro",
    mask_alpha: Optional[float] = 0,
) -> PolyData or UnstructuredGrid:
    """
    Add rgba color to each point of mesh based on labels.

    Args:
        mesh: A reconstructed mesh.
        labels: An array of labels of interest.
        key_added: The key under which to add the labels.
        colormap: Colors to use for plotting data.
        alphamap: The opacity of the color to use for plotting data.
        mask_color: Color to use for plotting mask information.
        mask_alpha: The opacity of the color to use for plotting mask information.
    Returns:
         A mesh, which contains the following properties:
            `mesh.point_data[key_added]`, the labels array;
            `mesh.point_data[f'{key_added}_rgba']`, the rgba colors of the labels.
    """

    cu_arr = np.unique(labels)
    cu_arr = np.sort(cu_arr, axis=0)
    cu_dict = {}

    # Set mask rgba.
    mask_ind = np.argwhere(cu_arr == "mask")
    if len(mask_ind) != 0:
        cu_arr = np.delete(cu_arr, mask_ind[0])
        cu_dict["mask"] = mpl.colors.to_rgba(mask_color, alpha=mask_alpha)

    cu_arr_num = cu_arr.shape[0]
    if cu_arr_num != (0,):
        # Set alpha.
        alpha_list = alphamap if isinstance(alphamap, list) else [alphamap] * cu_arr_num

        # Set raw rgba.
        if isinstance(colormap, list):
            raw_rgba_list = [mpl.colors.to_rgba(color) for color in colormap]
        elif colormap in list(mpl.colormaps):
            lscmap = mpl.cm.get_cmap(colormap)
            raw_rgba_list = [lscmap(i) for i in np.linspace(0, 1, cu_arr_num)]
        else:
            raw_rgba_list = [mpl.colors.to_rgba(colormap)] * cu_arr_num

        # Set new rgba.
        for t, c, a in zip(cu_arr, raw_rgba_list, alpha_list):
            cu_dict[t] = mpl.colors.to_rgba(c, alpha=a)

    # Added labels and rgba of the labels
    mesh.point_data[key_added] = labels
    mesh.point_data[f"{key_added}_rgba"] = np.array([cu_dict[g] for g in labels]).astype(np.float64)

    return mesh


def merge_mesh(
    meshes: List[PolyData or UnstructuredGrid],
) -> PolyData or UnstructuredGrid:
    """Merge all meshes in the `meshes` list. The format of all meshes must be the same."""

    merged_mesh = meshes[0]
    for mesh in meshes[1:]:
        merged_mesh.merge(mesh, inplace=True)

    return merged_mesh


def collect_mesh(
    meshes: List[PolyData or UnstructuredGrid],
    meshes_name: Optional[List[str]] = None,
) -> MultiBlock:
    """
    A composite class to hold many data sets which can be iterated over.
    You can think of MultiBlock like lists or dictionaries as we can iterate over this data structure by index
    and we can also access blocks by their string name.

    If the input is a dictionary, it can be iterated in the following ways:
        >>> blocks = collect_mesh(meshes, meshes_name)
        >>> for name in blocks.keys():
        ...     print(blocks[name])

    If the input is a list, it can be iterated in the following ways:
        >>> blocks = collect_mesh(meshes)
        >>> for block in blocks:
        ...    print(block)
    """

    if meshes_name is not None:
        meshes = {name: mesh for mesh, name in zip(meshes, meshes_name)}

    return pv.MultiBlock(meshes)


def save_mesh(
    mesh: DataSet,
    filename: str,
    binary: bool = True,
    texture: Union[str, np.ndarray] = None,
):
    """
    Save the pvvista/vtk mesh to vtk files.
    Args:
        mesh: A reconstructed mesh.
        filename: Filename of output file. Writer type is inferred from the extension of the filename.
        binary: If True, write as binary. Otherwise, write as ASCII. Binary files write much faster than ASCII and have a smaller file size.
        texture: Write a single texture array to file when using a PLY file.
                 Texture array must be a 3 or 4 component array with the datatype np.uint8.
                 Array may be a cell array or a point array, and may also be a string if the array already exists in the PolyData.
                 If a string is provided, the texture array will be saved to disk as that name.
                 If an array is provided, the texture array will be saved as 'RGBA'
    """

    if filename.endswith(".vtk"):
        mesh.save(filename=filename, binary=binary, texture=texture)
    else:
        raise ValueError(
            "\nFilename is wrong. This function is only available when saving vtk files."
            "\nPlease enter a filename ending with `.vtk`."
        )


def read_vtk(filename: str) -> DataSet:
    """
    Read vtk file.
    Args:
        filename: The string path to the file to read.
    Returns:
        Wrapped PyVista dataset.
    """

    return pv.read(filename)
