import gzip
import io

import anndata as ad
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from typing import Optional, Union


def read_lasso(filename: Optional[str] = None):

    file_format = filename.split(".")[-1]
    if file_format == "gz":
        with gzip.open(filename, 'rb') as input_file:
            with io.TextIOWrapper(input_file, encoding='utf-8') as dec:
                dec_list = [str(i).strip().split("\t") for i in dec.readlines()]
                lasso_data = pd.DataFrame(dec_list[1:], columns=dec_list[0], dtype=str)
    elif file_format in ["gem", "txt"]:
        lasso_data = pd.read_csv(filename, sep="\t")
    else:
        raise ValueError("The file format is wrong, the file format must be one of '.gz', '.txt' and '.gem'.")

    lasso_data["geneID"] = lasso_data.geneID.astype(str).str.strip('"')
    lasso_data["x"] = lasso_data["x"].astype(int)
    lasso_data["y"] = lasso_data["y"].astype(int)
    lasso_data["MIDCounts"] = lasso_data["MIDCounts"].astype(int)
    return lasso_data


def lasso2adata(data : Optional[pd.DataFrame] = None,
                binsize: Union[int, float] = None,
                slice: Optional[str] = None,
                z: Union[int, float] = None,
                z_gap: Union[int, float] = None,
                ):
    # get cell name

    data['x_ind'] = np.floor((data['x'].values - np.min(data['x'])) / binsize).astype(int)
    data['y_ind'] = np.floor((data['y'].values - np.min(data['y'])) / binsize).astype(int)

    data['x_centroid'] = data['x_ind'].values * binsize + binsize / 2
    data['y_centroid'] = data['y_ind'].values * binsize + binsize / 2

    data['cell_name'] = data['x_ind'].astype(str) + '_' + data['y_ind'].astype(str)

    uniq_cell, uniq_gene = data.cell_name.unique().tolist(), data.geneID.unique().tolist()

    cell_dict = dict(zip(uniq_cell, range(0, len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(0, len(uniq_gene))))

    data["csr_x_ind"] = data["cell_name"].map(cell_dict)
    data["csr_y_ind"] = data["geneID"].map(gene_dict)

    count_name = 'UMICount' if 'UMICount' in data.columns else 'MIDCounts'
    csr_mat = csr_matrix((data[count_name], (data["csr_x_ind"], data["csr_y_ind"])),
                         shape=(len(uniq_cell), len(uniq_gene)))

    all_coords = data[['x_centroid', 'y_centroid', 'x', 'y']].drop_duplicates(inplace=False)
    coords = all_coords[['x_centroid', 'y_centroid']].applymap(lambda coord: round(coord / binsize, 2)).values
    raw_coords = all_coords[['x', 'y']].values

    # var
    var = pd.DataFrame({"gene_short_name": uniq_gene})
    var.set_index("gene_short_name", inplace=True)

    # obs
    obs = pd.DataFrame({"cell_name": uniq_cell, "slice": [slice] * len(uniq_cell), "x": coords[:, 0],
                        "y": coords[:, 1], "z": [z * 2 * z_gap / binsize] * len(uniq_cell)})
    obs.set_index("cell_name", inplace=True)

    # obsm
    obsm = {"spatial": coords, "raw_spatial": raw_coords}

    adata = ad.AnnData(csr_mat, obs=obs.copy(), var=var.copy(), obsm=obsm.copy())

    return adata

file = "E:\BGI_Paper\L3_new\L3_b\lasso\lasso_L3_b_bin1\L3_b_S24_1.gem.gz"
data = read_lasso(filename=file)
adata = lasso2adata(data, slice="L3_b_S24", binsize=1, z=24, z_gap=4)

