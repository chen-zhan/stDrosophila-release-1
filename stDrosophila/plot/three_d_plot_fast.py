import re
import math
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib as mpl
import seaborn as sns
from anndata import AnnData
from typing import Union, Optional, List

def set_mesh(adata: AnnData,
             cluster: str = 'cluster',
             cluster_show: Union[str, list] = "all",
             gene_show: Union[str, list] = "all",
             colormap: Union[str, list, dict] = "viridis"
             ):
    '''

    Create a mesh.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        an Annodata object.
    cluster : `str`
        Column name in .obs DataFrame that stores clustering results.
    cluster_show : `str` or `list` (default: `all`)
        Clustering categories that need to be displayed.
    gene_show : `str` or `list` (default: `all`)
        Genes that need to be displayed.
    colormap : `str` or `list` or `dict` (default: `viridis`)
        Name of the Matplotlib colormap.
        You can also create a color list or dictionary to specify the color of a specific cluster.
        For example, to create a two color list you might specify ['green', 'red'];
        to create a two color dictionary you might specify cluster {'cluster1':'green', 'cluster2':'red'}.

    Returns
    -------
    clipped_grid:
        pyvista.PolyData
        Dataset consisting of clustering vertices.
    surf:
        pyvista.PolyData
        Clipped surface.
    colordict:
        A dictionary includes clusters and their colors.

    '''

    points = adata.obs[["x", "y", "z"]].values
    grid = pv.PolyData(points)

    # Add clusters' info
    if cluster_show == "all":
        grid["cluster"] = adata.obs[cluster]
        n_clusters = np.unique(grid["cluster"])
    elif isinstance(cluster_show, list) or isinstance(cluster_show, tuple):
        grid["cluster"] = adata.obs[cluster].map(lambda x: str(x) if x in cluster_show else "mask")
        n_clusters = np.array(cluster_show)
    else:
        grid["cluster"] = adata.obs[cluster].map(lambda x: x if x == cluster_show else "mask")
        n_clusters = np.array([cluster_show])

    # Set clusters' color
    if isinstance(colormap, list):
        colordict = {clu: col for clu, col in zip(n_clusters, colormap)}
    elif isinstance(colormap, dict):
        colordict = colormap
    else:
        cmap = [mpl.colors.to_hex(i, keep_alpha=False) for i in
                sns.color_palette(palette=colormap, n_colors=n_clusters.size, as_cmap=False)]
        colordict = {clu: col for clu, col in zip(n_clusters, cmap)}
    colordict['mask'] = "gainsboro"

    # Add genes' info
    if gene_show == "all":
        grid["gene"] = adata.X.sum(axis=1)
    else:
        grid["gene"] = adata[:, gene_show].X.sum(axis=1)

    surf = grid.delaunay_3d().extract_geometry()
    surf.subdivide(nsub=3, subfilter="loop", inplace=True)
    clipped_grid = grid.clip_surface(surf)
    clipped_grid["cluster_color"] = np.array([colordict[i] for i in clipped_grid["cluster"]])

    return clipped_grid, surf, colordict


def recon_3D(adata: AnnData,
             cluster: str = 'cluster',
             save: str = "3d.png",
             cluster_show: Union[str, list] = "all",
             gene_show: Union[str, list] = "all",
             show: str = "cluster",
             surf_show: bool = True,
             colormap: Union[str, list, dict] = "viridis",
             background_color: str = "black",
             other_color: str = "white",
             plotter_shape: Union[str, tuple, None] = None,
             off_screen: bool = True,
             window_size: Optional[List[int]] = None,
             cpos: Union[str, tuple, list, None] = None,
             legend_position: Optional[list] = None,
             legend_height: float = 0.3,
             legend_size: Optional[list] = None,
             view_up: Optional[list] = None,
             framerate: int = 15
             ):
    """

    Draw a 3D image that integrates all the slices through pyvista,
    and you can output a png image file, or a gif image file, or an MP4 video file.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        an Annodata object.
    cluster: `str`
        Column name in .obs DataFrame that stores clustering results.
    save: `str`
        If a str, save the figure. Infer the file type if ending on
        {".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff", ".gif", ".mp4"}.
    cluster_show: `str` or `list` (default: `all`)
        Clustering categories that need to be displayed.
    gene_show: `str` or `list` (default: `all`)
        Genes that need to be displayed.
    show: `str` (default: `cluster`)
        Display gene expression (`gene`) or clustering results (`cluster`).
    surf_show: `bool` (default: `True`)
        Display surface when True.
    colormap: `str` or `list` or `dict` (default: `viridis`)
        Name of the Matplotlib colormap.
        You can also create a color list or dictionary to specify the color of a specific cluster.
        For example, to create a three color list you might specify ["green", "red", "blue"];
        to create a three color dictionary you might specify {"cluster1":"green", "cluster2":"red", "cluster3":"blue"}.
    background_color: `str` (default: `black`)
        The background color of the active render window.
    other_color: `str` (default: `white`)
        The color of the font and border.
    plotter_shape: `str` or `tuple` (default: `None`)
        Number of sub-render windows inside of the main window. Specify two across with shape=(2, 1) and a two by two
        grid with shape=(2, 2).
        Can also accept a string descriptor as shape. E.g.:
        shape="3|1" means 3 plots on the left and 1 on the right,
        shape="4/2" means 4 plots on top and 2 at the bottom.
        Defaults to "3|1".
    off_screen: `bool` (default: `True`)
        Renders off screen when True. Useful for automated screenshots.
    window_size: `list` (optional, default: `None`)
        Window size in pixels. Defaults to [1024, 768].
    cpos: `str` or `tuple` or `list` (default: `None`)
        List of Camera position. Available camera positions are: "xy", "xz", "yz", "yx", "zx", "zy", "iso".
        Can also accept a tuple descriptor. E.g.: (7, 0, 20.).
        Defaults to ["xy", "xz", "yz", "iso"].
    legend_position: `list` (optional, default: `None`)
        The percentage (0 to 1) along the windows’s horizontal direction and vertical direction to place the bottom
        left corner of the legend. Defaults to [0.9, 0.1].
    legend_height: `float` (default: `0.3`)
        The percentage (0 to 1) height of the window for the colorbar.
    legend_size: `list` (optional, default: `None`)
        Two float list, each float between 0 and 1. For example [0.1, 0.1] would make the legend 10% the size of the
        entire figure window. Defaults to [0.9, 0.1].
    view_up: `list` (optional, default: `None`)
        The normal to the orbital plane. Defaults to [0.5, 0.5, 1].
    framerate: `int` (default: `15`)
        Frames per second.

    Examples
    --------
    >> adata
    AnnData object with n_obs × n_vars = 35145 × 16131
    obs: 'slice_ID', 'x', 'y', 'z', 'cluster'
    obsm: 'spatial'
    >> recon_3d(adata=adata, cluster="cluster", cluster_show=["muscle", "testis"], gene_show=["128up", "14-3-3epsilon"],
                show='cluster', save="3d.png", viewup=[0, 0, 0], colormap="RdYlBu_r", legend_height=0.2)

    """

    if plotter_shape is None:
        plotter_shape = "3|1"
    else:
        pass
    if window_size is None:
        window_size = [1024, 768]
    else:
        pass
    if cpos is None:
        cpos = ["xy", "xz", "yz", "iso"]
    elif isinstance(cpos, str) or isinstance(cpos, tuple):
        cpos = [cpos]
    else:
        pass
    if legend_position is None:
        legend_position = [0.9, 0.1]
    else:
        pass
    if legend_size is None:
        legend_size = [0.1, 0.1]
    else:
        pass

    # Create a mesh
    mesh, surf, colordict = set_mesh(adata=adata, cluster=cluster, cluster_show=cluster_show,
                                     gene_show=gene_show, colormap=colormap)

    mesh_data = pd.DataFrame(mesh.points)
    mesh_data["cluster"] = mesh["cluster"]
    mesh_data["cluster_color"] = mesh["cluster_color"]
    mesh_data["gene"] = mesh["gene"]

    other_data = mesh_data[mesh_data["cluster"] != "mask"]
    other_grid = pv.PolyData(other_data[[0, 1, 2]].values)
    other_grid["cluster"] = other_data["cluster"]
    other_grid["cluster_color"] = np.array([mpl.colors.to_rgb(i) for i in other_data["cluster_color"]])
    other_grid["gene"] = other_data["gene"]

    # Plotting object to display vtk meshes
    p = pv.Plotter(shape=plotter_shape, off_screen=off_screen, border=True, border_color=other_color,
                   lighting="light_kit", window_size=window_size)
    p.background_color = background_color
    if isinstance(plotter_shape, str):
        n = re.split("[|/]", plotter_shape)
        subplot_index_list = [i for i in range(int(n[0]) + int(n[1]))]
    else:
        subplot_index_list = [[i, j] for i in range(plotter_shape[0]) for j in range(plotter_shape[1])]

    for subplot_index, i, cpo in zip(subplot_index_list, range(len(cpos)), cpos):
        if isinstance(subplot_index, int):
            p.subplot(subplot_index)
        elif isinstance(subplot_index, list):
            p.subplot(subplot_index[0], subplot_index[1])
        # Add clipped surface
        if surf_show:
            p.add_mesh(surf, show_scalar_bar=False, show_edges=False, opacity=0.2, color="whitesmoke")
        # Add undisplayed clustering vertices
        if cluster_show != "all":
            mask_data = mesh_data[mesh_data["cluster"] == "mask"]
            mask_grid = pv.PolyData(mask_data[[0, 1, 2]].values)
            p.add_mesh(mask_grid, opacity=0.02, color="gainsboro", render_points_as_spheres=True, ambient=0.5)
        # Add displayed clustering vertices
        if show == "cluster":
            p.add_mesh(other_grid, opacity=0.7, scalars="cluster_color",  ambient=0.5, rgb=True, show_scalar_bar=False,
                       render_points_as_spheres=True)
            if int(i+1) == len(cpos):
                legend_entries = [[clu, col] for clu, col in colordict.items() if clu != "mask"]
                p.add_legend(legend_entries, face="circle", bcolor=None, origin=legend_position, size=legend_size)

        elif show == "gene":
            p.add_mesh(other_grid, opacity=0.7, scalars="gene", colormap=colormap, ambient=0.5,
                       render_points_as_spheres=True)
            p.remove_scalar_bar()
            if int(i+1) == len(cpos):
                p.add_scalar_bar(fmt="%.2f", font_family="arial", color=other_color, vertical=True, use_opacity=True,
                                 position_x=legend_position[0], position_y=legend_position[1], height=legend_height)
        p.show_axes()
        p.camera_position = cpo
        fontsize = math.ceil(window_size[0] / 100)
        p.add_text(f"\n "
                   f" Camera position = '{cpo}' \n "
                   f" Cluster(s): {cluster_show} \n "
                   f" Gene(s): {gene_show} ",
                   position="upper_left", font="arial", font_size=fontsize, color=mpl.colors.to_hex(other_color))

    # Save 3D reconstructed image or GIF or video
    if save.endswith(".png") or save.endswith(".tif") or save.endswith(".tiff") \
            or save.endswith(".bmp") or save.endswith(".jpeg") or save.endswith(".jpg"):
        p.show(screenshot=save)
    else:
        path = p.generate_orbital_path(factor=2.0, shift=0, viewup=view_up, n_points=20)
        if save.endswith(".gif"):
            p.open_gif(save)
        elif save.endswith(".mp4"):
            p.open_movie(save, framerate=framerate, quality=5)
        p.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)
        p.close()

def plane_3D(adata: AnnData,
             cluster: str = 'cluster',
             save: str = "3d.png",
             cluster_show: Union[str, list] = "all",
             gene_show: Union[str, list] = "all",
             show: str = "cluster",
             colormap: Union[str, list, dict] = "viridis",
             background_color: str = "black",
             other_color: str = "white",
             plotter_shape: Union[str, tuple, None] = None,
             off_screen: bool = True,
             window_size: Optional[List[int]] = None,
             cpos: Union[str, tuple, list, None] = None,
             clip_cpos: Union[str, tuple, list, None] = None,
             legend_position: Optional[list] = None,
             legend_height: float = 0.3,
             legend_size: Optional[list] = None,
             ):
    """

    Reconstruct the 3D model, and cut off the 3D model to obtain its axial plane.

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        an Annodata object.
    cluster : `str`
        Column name in .obs DataFrame that stores clustering results.
    save : `str`
        If a str, save the figure. Infer the file type if ending on
        {".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff", ".gif", ".mp4"}.
    cluster_show : `str` or `list` (default: `all`)
        Clustering categories that need to be displayed.
    gene_show : `str` or `list` (default: `all`)
        Genes that need to be displayed.
    show : `str` (default: `cluster`)
        Display gene expression (`gene`) or clustering results (`cluster`).
    colormap : `str` or `list` or `dict` (default: `viridis`)
        Name of the Matplotlib colormap.
        You can also create a color list or dictionary to specify the color of a specific cluster.
        For example, to create a three color list you might specify ["green", "red", "blue"];
        to create a three color dictionary you might specify {"cluster1":"green", "cluster2":"red", "cluster3":"blue"}.
    background_color : `str` (default: `black`)
        The background color of the active render window.
    other_color : `str` (default: `white`)
        The color of the font and border.
    plotter_shape : `str` or `tuple` (default: `None`)
        Number of sub-render windows inside of the main window. Specify two across with shape=(2, 1) and a two by two
        grid with shape=(2, 2).
        Can also accept a string descriptor as shape. E.g.:
        shape="3|1" means 3 plots on the left and 1 on the right,
        shape="4/2" means 4 plots on top and 2 at the bottom.
        Defaults to (1, 3).
    off_screen : `bool` (default: `True`)
        Renders off screen when True. Useful for automated screenshots.
    window_size : `list` (optional, default: `None`)
        Window size in pixels. Defaults to [1024, 768].
    cpos : `str` or `tuple` or `list` (default: `None`)
        List of Camera position. Available camera positions are: "xy", "xz", "yz", "yx", "zx", "zy", "iso".
        Can also accept a tuple descriptor. E.g.: (7, 0, 20.).
        Defaults to ["xy", "xz", "yz"].
    clip_cpos : `str` or `tuple` or `list` (default: `None`)
        Length 3 tuple for the normal vector direction.
        Can also be specified as a string conventional direction such as 'x' for (1,0,0) or '-x' for (-1,0,0), etc.
        Defaults to ["-z", "y", "-x"].
    legend_position : `list` (optional, default: `None`)
        The percentage (0 to 1) along the windows’s horizontal direction and vertical direction to place the bottom
        left corner of the legend. Defaults to [0.9, 0.1].
    legend_height : `float` (default: `0.3`)
        The percentage (0 to 1) height of the window for the colorbar.
    legend_size : `list` (optional, default: `None`)
        Two float list, each float between 0 and 1. For example [0.1, 0.1] would make the legend 10% the size of the
        entire figure window. Defaults to [0.9, 0.1].

    Examples
    --------
    #>>> adata
    AnnData object with n_obs × n_vars = 35145 × 16131
    obs: 'slice_ID', 'x', 'y', 'z', 'cluster'
    obsm: 'spatial'
    #>>> plane_3D(adata=adata,
                  cluster="anno",
                  cluster_show=['fat body', 'muscle', 'testis'],
                  gene_show=['RpS12'],
                  show='cluster',
                  off_screen=False,
                  save='plane.png'
                  )

    """

    if plotter_shape is None:
        plotter_shape = (1, 3)
    else:
        pass
    if window_size is None:
        window_size = [1024, 768]
    else:
        pass
    if cpos is None:
        cpos = ["xy", "xz", "yz"]
    elif isinstance(cpos, str) or isinstance(cpos, tuple):
        cpos = [cpos]
    else:
        pass
    if clip_cpos is None:
        clip_cpos = ["-z", "y", "-x"]
    elif isinstance(clip_cpos, str) or isinstance(clip_cpos, tuple):
        clip_cpos = [clip_cpos]
    else:
        pass
    if legend_position is None:
        legend_position = [0.9, 0.1]
    else:
        pass
    if legend_size is None:
        legend_size = [0.1, 0.1]
    else:
        pass

    # Create a mesh
    mesh, surf, colordict = set_mesh(adata=adata, cluster=cluster, cluster_show=cluster_show,
                                             gene_show=gene_show, colormap=colormap)

    # Clip a dataset by a plane by specifying the origin and normal.
    clip_mesh_list = []
    for clip_cpo in clip_cpos:
        clip_mesh = mesh.clip(clip_cpo, invert=False)
        clip_mesh_data = pd.DataFrame(clip_mesh.points)
        clip_mesh_data["cluster"] = clip_mesh["cluster"]
        clip_mesh_data["cluster_color"] = clip_mesh["cluster_color"]
        clip_mesh_data["gene"] = clip_mesh["gene"]

        clip_other_data = clip_mesh_data[clip_mesh_data["cluster"] != "mask"]
        clip_other = pv.PolyData(clip_other_data[[0, 1, 2]].values)
        clip_other["cluster"] = clip_other_data["cluster"]
        clip_other["cluster_color"] = np.array([mpl.colors.to_rgb(i) for i in clip_other_data["cluster_color"]])
        clip_other["gene"] = clip_other_data["gene"]
        clip_mesh_list.append([clip_mesh_data, clip_other])

    # Plotting object to display vtk meshes
    p = pv.Plotter(shape=plotter_shape, off_screen=off_screen, border=True, border_color=other_color,
                   lighting="light_kit", window_size=window_size)
    p.background_color = background_color
    if isinstance(plotter_shape, str):
        n = re.split("[|/]", plotter_shape)
        subplot_index_list = [i for i in range(int(n[0]) + int(n[1]))]
    else:
        subplot_index_list = [[i, j] for i in range(plotter_shape[0]) for j in range(plotter_shape[1])]

    for subplot_index, i, cpo, clip_meshes in zip(subplot_index_list, range(len(cpos)), cpos, clip_mesh_list):
        if isinstance(subplot_index, int):
            p.subplot(subplot_index)
        elif isinstance(subplot_index, list):
            p.subplot(subplot_index[0], subplot_index[1])

        # Add clipped surface
        p.add_mesh(surf, show_scalar_bar=False, show_edges=False, opacity=0.2, color="whitesmoke")

        # Add undisplayed clustering vertices
        if cluster_show != "all":
            clip_mesh_data = clip_meshes[0]
            mask_data = clip_mesh_data[clip_mesh_data["cluster"] == "mask"]
            mask_mesh = pv.PolyData(mask_data[[0, 1, 2]].values)
            p.add_mesh(mask_mesh, opacity=0.02, color="gainsboro", render_points_as_spheres=True, ambient=0.5)

        # Add displayed clustering vertices
        if show == "cluster":
            p.add_mesh(clip_meshes[1], opacity=0.7, scalars="cluster_color", ambient=0.5, rgb=True,
                       show_scalar_bar=False, render_points_as_spheres=True)
            if int(i+1) == len(cpos):
                legend_entries = [[clu, col] for clu, col in colordict.items() if clu != "mask"]
                print(legend_entries)
                p.add_legend(legend_entries, face="circle", bcolor=None, origin=legend_position, size=legend_size)
        elif show == "gene":
            p.add_mesh(clip_meshes[1], opacity=0.7, scalars="gene", colormap=colormap, ambient=0.5,
                       render_points_as_spheres=True)
            p.remove_scalar_bar()
            if int(i+1) == len(cpos):
                p.add_scalar_bar(fmt="%.2f", font_family="arial", color=other_color, vertical=True, use_opacity=True,
                                 position_x=legend_position[0], position_y=legend_position[1], height=legend_height)

        p.show_axes()
        p.camera_position = cpo
        fontsize = math.ceil(window_size[0] / 100)
        p.add_text(f"\n "
                   f" Camera position = '{cpo}' \n "
                   f" Cluster(s): {cluster_show} \n "
                   f" Gene(s): {gene_show} ",
                   position="upper_left", font="arial", font_size=fontsize, color=mpl.colors.to_hex(other_color))

    # Save slices image from a mesh
    p.show(screenshot=save)