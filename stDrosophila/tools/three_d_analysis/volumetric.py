import numpy as np
import pandas as pd
import pyvista as pv

from pyvista import PolyData, UnstructuredGrid
from typing import Optional, Union


def compute_volume(
    mesh: Union[PolyData, UnstructuredGrid]
) -> float:
    """
    Calculate the volume of the reconstructed 3D structure.
    Args:
        mesh: Reconstructed 3D structure (voxelized object).
    Returns:
        volume_size: The volume of the reconstructed 3D structure.
    """

    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)
    volume_size = float(np.sum(mesh.cell_data["Volume"]))
    print(f"volume: {volume_size}")

    return volume_size

"""
def compute_area():
"""