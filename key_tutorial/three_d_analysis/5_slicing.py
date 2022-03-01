import pyvista as pv
import stDrosophila as sd

surf = pv.read(r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\merge_surf.vtk")
slices_a = sd.tl.three_d_slice(mesh=surf, key="groups", slice_method="axis")

slices_o = sd.tl.three_d_slice(mesh=surf, key="groups", slice_method="orthogonal")
