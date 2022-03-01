import pyvista as pv
import numpy as np
import matplotlib as mpl
import stDrosophila as sd

fb_pcd = pv.read(r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\fb_pcd.vtk")
fb_surf = pv.read(r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\fb_surf.vtk")
cns_pcd = pv.read(
    r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\cns_pcd_new.vtk"
)
cns_surf = pv.read(
    r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\cns_surf_new.vtk"
)
body_surf = pv.read(
    r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\basic_surf.vtk"
)
print(fb_pcd.points)
pcd = sd.tl.merge_mesh([fb_pcd, cns_pcd])
surf = sd.tl.merge_mesh([fb_surf, cns_surf, body_surf])

sd.pl.three_d_plot(
    mesh=pcd,
    key="groups",
    off_screen=False,
    window_size=(1024, 768),
    background="white",
    background_r="black",
    ambient=0.3,
    opacity=1.0,
    initial_cpo="iso",
    legend_loc="lower right",
    legend_size=(0.1, 0.1),
    filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_img\merge_pcd.png",
    view_up=(0.5, 0.5, 1),
    framerate=15,
    plotter_filename=None,
)
sd.pl.three_d_plot(
    mesh=surf,
    key="groups",
    off_screen=False,
    window_size=(1024, 768),
    background="white",
    background_r="black",
    ambient=0.3,
    opacity=1.0,
    initial_cpo="iso",
    legend_loc="lower right",
    legend_size=(0.1, 0.1),
    filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_img\merge_surf.png",
    view_up=(0.5, 0.5, 1),
    framerate=15,
    plotter_filename=None,
)

p = pv.Plotter()
p.background_color = "white"
p.add_mesh(pcd)
p.add_mesh(surf)
p.show(screenshot=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_img\all.png")

sd.tl.mesh_to_vtk(
    mesh=surf,
    filename=r"D:\BGIpy37_pytorch113\three_d_analysis_image\mesh_vtk\merge_surf.vtk",
)
