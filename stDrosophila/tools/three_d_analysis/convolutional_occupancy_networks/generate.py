import torch
import numpy as np
import os
import shutil

from src import config
from src.checkpoints import CheckpointIO

config_file = '/home/yao/BGIpy37_pytorch113/convolutional_occupancy_networks/configs/pointcloud_crop/demo_matterport.yaml'

cfg = config.load_config(config_file, 'configs/default.yaml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(model=model)
checkpoint_io.load(cfg['test']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Determine what to generate
generate_mesh = cfg['generation']['generate_mesh']
generate_pointcloud = cfg['generation']['generate_pointcloud']

if generate_mesh and not hasattr(generator, 'generate_mesh'):
    generate_mesh = False
    print('Warning: generator does not support mesh generation.')

if generate_pointcloud and not hasattr(generator, 'generate_pointcloud'):
    generate_pointcloud = False
    print('Warning: generator does not support pointcloud generation.')

# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=0, shuffle=False)

# Generate
model.eval()
data = [i for i in test_loader][0]
if generate_mesh:
    if cfg['generation']['sliding_window']:
        out = generator.generate_mesh_sliding(data)
    else:
        out = generator.generate_mesh(data)
    try:
        mesh, stats_dict = out
    except TypeError:
        mesh, stats_dict = out, {}
    print(mesh)
    vertices = mesh.vertices.astype(np.float64)
    print(vertices)
    faces = mesh.faces
    face_type = np.full((faces.shape[0], 1), faces.shape[1])
    faces = np.concatenate((face_type, faces), axis=1).flatten().astype(np.int64)
    print(faces)

if generate_pointcloud:
    pointcloud = generator.generate_pointcloud(data)
    print(pointcloud)

if os.path.exists('./chkpts'):
    shutil.rmtree('./chkpts')

import pyvista as pv
surf = pv.PolyData(vertices, faces)
surf.plot()