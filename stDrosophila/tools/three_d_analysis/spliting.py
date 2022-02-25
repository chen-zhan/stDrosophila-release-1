import warnings

import numpy as np
import pyvista as pv
import vtk

from pyvista.core.pointset import PolyData, UnstructuredGrid
from typing import Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def three_d_split(
    mesh: Union[PolyData, UnstructuredGrid],
):
    if isinstance(mesh, UnstructuredGrid) is False:
        warnings.warn(
            "The mesh should be a pyvista.UnstructuredGrid object."
        )
        mesh = mesh.cast_to_unstructured_grid()

    p = pv.Plotter()
    p.add_mesh(mesh, color='w')

    picked = []
    legend = []

    def split_mesh(original_mesh):
        """Adds a new mesh to the plotter each time cells are picked, and
        removes them from the original mesh"""

        # if nothing selected
        if not original_mesh.n_cells:
            return

        # remove the picked cells from main grid
        ghost_cells = np.zeros(mesh.n_cells, np.uint8)
        ghost_cells[original_mesh['orig_extract_id']] = 1
        mesh.cell_data[vtk.vtkDataSetAttributes.GhostArrayName()] = ghost_cells
        mesh.RemoveGhostCells()

        # add the selected mesh this to the main plotter
        color = np.random.random(3)
        legend.append(['picked mesh %d' % len(picked), color])
        p.add_mesh(original_mesh, color=color)
        p.add_legend(legend)

        # track the picked meshes and label them
        original_mesh['picked_index'] = np.ones(original_mesh.n_points) * len(picked)
        picked.append(original_mesh)

    # enable cell picking with our custom callback
    p.enable_cell_picking(mesh=mesh, callback=split_mesh, show=False)
    p.show()

    # convert these meshes back to surface meshes (PolyData)
    separated_meshes = []
    for selected_mesh in picked:
        separated_meshes.append(selected_mesh.extract_surface())
    # pv.plot(separated_meshes)

    return separated_meshes
