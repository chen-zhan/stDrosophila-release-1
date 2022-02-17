import math
import re

import matplotlib as mpl
import numpy as np
import pandas as pd
import pyvista as pv

from typing import Optional, Sequence, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def easy_three_d_plot(
    mesh: Optional[pv.DataSet] = None,
    scalar: Literal["groups", "genes"] = "groups",
    outline: bool = False,
    ambient: float = 0.3,
    opacity: float = 0.5,
    background: str = "black",
    background_r: str = "white",
    save: Optional[str] = None,
    notebook: bool = False,
    shape: Optional[list] = None,
    off_screen: bool = False,
    window_size: Optional[list] = None,
    cpos: Union[str, tuple, list] = "iso",
    legend_loc: Literal["upper right", "upper left",
                        "lower left", "lower right",
                        "center left", "lower center",
                        "upper center", "center"] = "lower right",
    legend_size: Optional[Sequence] = None,
    view_up: Optional[list] = None,
    framerate: int = 15,
):
    """
    Create a plotting object to display pyvista/vtk mesh.
    Args:
        mesh: Reconstructed 3D structure.
        scalar: Types used to “color” the mesh. Available scalars are:
                * `'groups'`
                * `'genes'`
        outline: Produce an outline of the full extent for the input dataset.
        ambient: When lighting is enabled, this is the amount of light in the range of 0 to 1 (default 0.0) that reaches
                 the actor when not directed at the light source emitted from the viewer.
        opacity: Opacity of the mesh. If a single float value is given, it will be the global opacity of the mesh and
                 uniformly applied everywhere - should be between 0 and 1.
                 A string can also be specified to map the scalars range to a predefined opacity transfer function
                 (options include: 'linear', 'linear_r', 'geom', 'geom_r').
        background: The background color of the window.
        background_r: A color that is clearly different from the background color.
        save: If a str, save the figure. Infer the file type if ending on
             {".png", ".jpeg", ".jpg", ".bmp", ".tif", ".tiff", ".gif", ".mp4"}.
        notebook: When True, the resulting plot is placed inline a jupyter notebook.
        shape: Number of sub-render windows inside of the main window. Available shape formats are:
                * `shape=(2, 1)`: 2 plots in two rows and one column
                * `shape='3|1'`: 3 plots on the left and 1 on the right,
                * `shape='4/2'`: 4 plots on top and 2 at the bottom.
        off_screen: Renders off screen when True. Useful for automated screenshots.
        window_size: Window size in pixels. The default window_size is `[1024, 768]`.
        cpos: Camera position of the window. Available cpos are:
                * `'xy'`, `'xz'`, `'yz'`, `'yx'`, `'zx'`, `'zy'`, `'iso'`
                * Customize a tuple. E.g.: (7, 0, 20.).
        legend_loc: The location of the legend in the window. Available legend_loc are:
                * `'upper right'`
                * `'upper left'`
                * `'lower left'`
                * `'lower right'`
                * `'center left'`
                * `'lower center'`
                * `'upper center'`
                * `'center'`
        legend_size: The size of the legend in the window. Two float sequence, each float between 0 and 1.
                     E.g.: (0.1, 0.1) would make the legend 10% the size of the entire figure window.
        view_up: The normal to the orbital plane. The default view_up is `[0.5, 0.5, 1]`.
        framerate: Frames per second.
    """

    if shape is None:
        shape = (1, 1)

    if isinstance(shape, str):
        n = re.split("[|/]", shape)
        subplot_indices = [i for i in range(int(n[0]) + int(n[1]))]
    else:
        subplot_indices = [[i, j] for i in range(shape[0]) for j in range(shape[1])]

    if window_size is None:
        window_size = (1024, 768)

    if type(cpos) in [str, tuple]:
        cpos = [cpos]

    if len(cpos) != len(subplot_indices):
        raise ValueError(
            "The number of cpos does not match the number of subplots drawn."
        )

    # Create a plotting object to display pyvista/vtk mesh.
    p = pv.Plotter(
        shape=shape,
        off_screen=off_screen,
        lighting="light_kit",
        window_size=window_size,
        notebook=notebook,
        border=True,
        border_color=background_r,
    )
    for subplot_index, cpo in zip(subplot_indices, cpos):

        # Add a reconstructed 3D structure.
        p.add_mesh(
            mesh,
            scalars=f"{scalar}_rgba",
            rgba=True,
            render_points_as_spheres=True,
            ambient=ambient,
            opacity=opacity,
        )

        # Add a legend to render window.
        mesh[f"{scalar}_hex"] = np.array(
            [mpl.colors.to_hex(i) for i in mesh[f"{scalar}_rgba"]]
        )
        _data = pd.concat(
            [pd.Series(mesh[scalar]), pd.Series(mesh[f"{scalar}_hex"])], axis=1
        )
        _data.columns = ["label", "hex"]
        _data = _data[_data["label"] != "mask"]
        _data.drop_duplicates(inplace=True)
        _data.sort_values(by=["label", "hex"], inplace=True)
        _data = _data.astype(str)
        gap = math.ceil(len(_data.index) / 5) if scalar is "genes" else 1
        legend_entries = [
            [_data["label"].iloc[i], _data["hex"].iloc[i]]
            for i in range(0, len(_data.index), gap)
        ]
        if scalar is "genes":
            legend_entries.append([_data["label"].iloc[-1], _data["hex"].iloc[-1]])

        legend_size = (0.1, 0.1) if legend_size is None else legend_size
        p.add_legend(
            legend_entries,
            face="circle",
            bcolor=None,
            loc=legend_loc,
            size=legend_size,
        )

        if outline:
            p.add_mesh(mesh.outline(), color=background_r, line_width=3)

        p.camera_position = cpo
        p.background_color = background
        p.show_axes()

    # Save as image or gif or mp4
    save = "three_d_structure.jpg" if save is None else save
    save_format = save.split(".")[-1]
    if save_format in ["png", "tif", "tiff", "bmp", "jpeg", "jpg"]:
        p.show(screenshot=save)
    else:
        view_up = [0.5, 0.5, 1] if view_up is None else view_up
        path = p.generate_orbital_path(factor=2.0, shift=0, viewup=view_up, n_points=20)
        if save.endswith(".gif"):
            p.open_gif(save)
        elif save.endswith(".mp4"):
            p.open_movie(save, framerate=framerate, quality=5)
        p.orbit_on_path(path, write_frames=True, viewup=(0, 0, 1), step=0.1)
        p.close()
