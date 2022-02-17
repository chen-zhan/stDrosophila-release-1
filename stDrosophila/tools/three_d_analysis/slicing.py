import warnings

import pyvista as pv

from pyvista.core.pointset import PolyData, UnstructuredGrid
from typing import Optional, Sequence, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def three_d_slicing(
    mesh: UnstructuredGrid,
    scalar: Literal["groups", "genes"] = "groups",
    axis: Union[str, int] = "x",
    n_slices: Union[str, int] = 10,
    center: Optional[Sequence[float]] = None,
) -> PolyData:
    """
    Create many slices of the input dataset along a specified axis or
    create three orthogonal slices through the dataset on the three cartesian planes.
    Args:
        mesh: Reconstructed 3D structure (voxelized object).
        scalar: Types used to “color” the mesh. Available scalars are:
                * `'groups'`
                * `'genes'`
        axis: The axis to generate the slices along. Available axes are:
                * `'x'` or `0`
                * `'y'` or `1`
                * `'z'` or `2`
        n_slices: The number of slices to create along a specified axis.
                  If n_slices is `"orthogonal"`, create three orthogonal slices.
        center: A 3-length sequence specifying the position which slices are taken. Defaults to the center of the mesh.
    Returns:
        Sliced dataset.
    """

    if isinstance(mesh, pv.core.pointset.UnstructuredGrid) is False:
        warnings.warn(
            "The model should be a pyvista.UnstructuredGrid (voxelized) object."
        )
        mesh = mesh.cast_to_unstructured_grid()

    mesh.set_active_scalars(f"{scalar}_rgba")

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
