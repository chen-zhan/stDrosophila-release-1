import numpy as np
import pyvista as pv

from pyvista import DataSet
from typing import Union


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


def read_vtk(
    filename: str
) -> DataSet:
    """
    Read vtk file.
    Args:
        filename: The string path to the file to read.
    Returns:
        Wrapped PyVista dataset.
    """

    return pv.read(filename)


