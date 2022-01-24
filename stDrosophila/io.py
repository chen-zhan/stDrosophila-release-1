import gzip
import io

import anndata as ad
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from typing import Optional, Union


def read_lasso(
        file: Optional[str] = None
) -> pd.DataFrame:

    lasso_data = pd.read_csv(file, sep="\t")
    lasso_data["geneID"] = lasso_data.geneID.astype(str).str.strip('"')
    lasso_data["x"] = lasso_data["x"].astype(int)
    lasso_data["y"] = lasso_data["y"].astype(int)
    count_name = "UMICount" if "UMICount" in lasso_data.columns else "MIDCounts"
    lasso_data[count_name] = lasso_data[count_name].astype(int)
    if "cell_type" in lasso_data.columns:
        lasso_data["cell_type"] = lasso_data["cell_type"].astype(str)

    return lasso_data


def lasso2adata(
        data: Optional[pd.DataFrame] = None,
        slice: Optional[str] = None,
        z: Union[int, float] = None,
        z_gap: Union[int, float] = None,
        physical_coords: bool = True
) -> ad.AnnData:

    data['x_ind'] = data['x'] - np.min(data['x'])
    data['y_ind'] = data['y'] - np.min(data['y'])

    data['cell_name'] = data['x_ind'].astype(str) + '_' + data['y_ind'].astype(str)

    uniq_cell, uniq_gene = data.cell_name.unique().tolist(), data.geneID.unique().tolist()

    cell_dict = dict(zip(uniq_cell, range(0, len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(0, len(uniq_gene))))

    data["csr_x_ind"] = data["cell_name"].map(cell_dict)
    data["csr_y_ind"] = data["geneID"].map(gene_dict)
    count_name = "UMICount" if "UMICount" in data.columns else "MIDCounts"
    csr_mat = csr_matrix((data[count_name], (data["csr_x_ind"], data["csr_y_ind"])),
                         shape=(len(uniq_cell), len(uniq_gene)))

    all_coords_col = ["x_ind", "y_ind", "x", "y", "cell_type"] if "cell_type" in data.columns else ["x_ind", "y_ind", "x", "y"]
    all_coords = data[all_coords_col].drop_duplicates(inplace=False)
    coords = all_coords[["x_ind", "y_ind"]].values * 0.5 if physical_coords else all_coords[["x_ind", "y_ind"]].values
    raw_coords = all_coords[["x", "y"]].values

    # var
    var = pd.DataFrame({"gene_short_name": uniq_gene})
    var.set_index("gene_short_name", inplace=True)

    # obs
    obs = pd.DataFrame({"cell_name": uniq_cell, "slice": [slice] * len(uniq_cell), "x": coords[:, 0], "y": coords[:, 1]})
    if z is not None and z_gap is not None:
        obs["z"] = [z * z_gap] * len(uniq_cell) if physical_coords else [z * z_gap * 2] * len(uniq_cell)
    if "cell_type" in data.columns:
        obs["cell_type"] = all_coords["cell_type"].to_numpy()
    obs.set_index("cell_name", inplace=True)

    # obsm
    obsm = {"spatial": coords, "raw_spatial": raw_coords}

    adata = ad.AnnData(csr_mat, obs=obs.copy(), var=var.copy(), obsm=obsm.copy())

    return adata
