import pyvista as pv
import numpy as np
import matplotlib as mpl
import stDrosophila as sd


new_pcd = pv.read(r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\cns_pcd.vtk")
pcd = sd.tl.three_d_pick(mesh=new_pcd, key="groups", merge=True, invert=True)
pcd.plot(screenshot=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_img\cns_pcd_new.png", background="white")

key_added = "groups"
mesh_color="skyblue"
mesh_alpha=0.5
mesh = sd.tl.construct_surface(pcd=pcd, cs_method="basic", surface_smoothness=0,
                               cs_method_args={"al_alpha": 10}, n_surf=50000, mtype="unstructured")
mesh.point_data[key_added] = np.array(["mask"] * mesh.n_points).astype(str)
mesh.point_data[f"{key_added}_rgba"] = np.array(
    [mpl.colors.to_rgba(mesh_color, alpha=mesh_alpha)] * mesh.n_points
).astype(np.float64)

mesh.cell_data[key_added] = np.array(["mask"] * mesh.n_cells).astype(str)
mesh.cell_data[f"{key_added}_rgba"] = np.array(
    [mpl.colors.to_rgba(mesh_color, alpha=mesh_alpha)] * mesh.n_cells
).astype(np.float64)
mesh.plot(screenshot=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_img\cns_surf_new.png", background="white")


sd.tl.mesh_to_vtk(mesh=pcd, filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\cns_pcd_new.vtk")
sd.tl.mesh_to_vtk(mesh=mesh, filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\cns_surf_new.vtk")


