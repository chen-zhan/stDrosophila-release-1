import warnings

import numpy as np
import pyvista as pv
import vtk

from pyvista import PolyData, UnstructuredGrid
from typing import List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def three_d_split(
    mesh: Union[PolyData, UnstructuredGrid],
    key: str = "groups",
) -> List[PolyData or UnstructuredGrid]:
    """
    Select the interested part of the reconstructed 3D mesh by interactive approach.
    Args:
        mesh: Reconstructed 3D mesh.
        key: The key under which are the labels.
    Returns:
        A list of multiple meshes.
    """

    if isinstance(mesh, UnstructuredGrid) is False:
        warnings.warn("The mesh should be a pyvista.UnstructuredGrid object.")
        mesh = mesh.cast_to_unstructured_grid()

    p = pv.Plotter()
    p.add_mesh(mesh, scalars=f"{key}_rgba", rgba=True)

    picked_meshes, legend = [], []

    def split_mesh(original_mesh):
        """Adds a new mesh to the plotter each time cells are picked, and
        removes them from the original mesh"""

        # if nothing selected
        if not original_mesh.n_cells:
            return

        # remove the picked cells from main grid
        ghost_cells = np.zeros(mesh.n_cells, np.uint8)
        ghost_cells[original_mesh["orig_extract_id"]] = 1
        mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = ghost_cells
        mesh.RemoveGhostCells()

        # add the selected mesh this to the main plotter
        color = np.random.random(3)
        legend.append(["picked mesh %d" % len(picked_meshes), color])
        p.add_mesh(original_mesh, color=color)
        p.add_legend(legend)

        # track the picked meshes and label them
        original_mesh["picked_index"] = np.ones(original_mesh.n_points) * len(
            picked_meshes
        )
        picked_meshes.append(original_mesh)

    # enable cell picking with our custom callback
    p.enable_cell_picking(mesh=mesh, callback=split_mesh, show=False)
    p.show()

    return picked_meshes
