
import random

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from scipy.sparse import csr_matrix, spmatrix
from .STAGATE_pyG import utils
from .STAGATE_pyG import STAGATE_pyG
from tqdm import tqdm


def STAGATE_SNN(adata, type="3D", slice_col="slice", cutoff_2D=50, cutoff_Zaxis=5,
                n_epoch=1000, lr=1e-3, weight_decay=1e-4, device=torch.device('cuda:0'), verbose=True):

    cudnn.deterministic = True
    cudnn.benchmark = True
    seed = 666
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if type == "2D":
        utils.Cal_Spatial_Net(adata, rad_cutoff=cutoff_2D, k_cutoff=None, model='Radius', verbose=verbose)
        utils.Stats_Spatial_Net(adata)
    elif type == "3D":
        slices_id = adata.obs[slice_col].unique().tolist()
        utils.Cal_Spatial_Net_3D(adata=adata,
                                 rad_cutoff_2D=cutoff_2D,
                                 rad_cutoff_Zaxis=cutoff_Zaxis,
                                 key_section=slice_col,
                                 section_order=slices_id,
                                 verbose=verbose)

    data = utils.Transfer_pytorch_Data(adata)
    model = STAGATE_pyG.STAGATE(in_channels=data.x.shape[1]).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in tqdm(range(1, n_epoch + 1)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        loss = F.mse_loss(data.x, out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

    model.eval()
    z, out = model(data.x, data.edge_index)
    STAGATE_rep = z.to('cpu').detach().numpy()
    adata.obsm['STAGATE'] = STAGATE_rep

    return adata