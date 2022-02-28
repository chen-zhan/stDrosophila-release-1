import pyvista
from pyvista import Plotter
import re

mesh = pyvista.Cube()
plotter = pyvista.Plotter()
plotter.add_mesh(mesh, scalars=range(6), show_scalar_bar=False)
plotter.add_camera_orientation_widget()
cpos = plotter.show(return_cpos=True)

plotter = pyvista.Plotter(off_screen=True)
plotter.add_mesh(mesh, scalars=range(6), show_scalar_bar=False)
plotter.add_camera_orientation_widget()

plotter.export_obj("sss.obj")


cpo, img = plotter.show(screenshot="sss.png", return_img=True, cpos=cpos, return_cpos=True)
print(cpo)
print(img)
