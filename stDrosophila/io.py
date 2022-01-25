import gzip
import io

import anndata as ad
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd

from anndata import AnnData
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from skimage import measure
from typing import Optional, Tuple, Union


def read_lasso(path: str) -> pd.DataFrame:

    lasso_data = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "geneID": "category",
            "x": np.uint32,
            "y": np.uint32,
            "MIDCounts": np.uint16,
            "cell": str,
        })

    lasso_data["geneID"] = lasso_data.geneID.astype(str).str.strip('"')

    return lasso_data


def lasso2adata(
    data: pd.DataFrame,
    slice: Optional[str] = None,
    label_path: Optional[str] = None,
    DNB_gap: Optional[float] = 0.5,
    z: Union[float] = None,
    z_gap: Union[float] = None,
    cellbin: bool = False
) -> AnnData:
    """A helper function that facilitates constructing an AnnData object suitable for downstream spateo analysis

    Parameters
    ----------
        data: `pandas.DataFrame`
            Lasso data.
        slice: `str` or `None` (default: `None`)
            Name of the slice. Will be used when displaying multiple slices.
        label_path: `str` or `None` (default: `None`)
            A string that points to the directory and filename of cell segmentation label matrix(`.npy` or `.npy.gz`).
        DNB_gap: `float` (default: `0.5`)
            True physical distance (microns) between nano balls.
        z: `float` (default: `None`)
            Z-axis direction coordinates.
        z_gap: `float` (default: `None`)
            True physical distance (microns) between slices.
        cellbin: `bool` (default: `False`)
            Whether to use cell bin as the base unit. Only valid when label_path is not None.
    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An AnnData object. Each row of the AnnData object correspond to a spot (aggregated with multiple bins). The
            `spatial` key in the .obsm corresponds to the x, y coordinates of the centroids of all spot.
    """
    # physical coords
    data["x"] = (data["x"].values - data["x"].values.min()) * DNB_gap
    data["y"] = (data["y"].values - data["y"].values.min()) * DNB_gap
    data["z"] = z * z_gap / DNB_gap if z is not None else 0

    # obs
    if label_path is not None:
        # TODO: Get cell names using labels
        if label_path.endswith(".gz"):
            with gzip.open(label_path, "r") as f:
                label_mtx = np.load(f)
        else:
            label_mtx = np.load(label_path)

        props = measure.regionprops_table(label_mtx, properties=("label", "centroid"))
        label_props = pd.DataFrame(props)
        label_props.columns = ["cell", "centroid_x", "centroid_y"]
        label_props["cell"] = label_props["cell"].astype(str)
        label_props["centroid_x"] = label_props["centroid_x"].values * DNB_gap
        label_props["centroid_y"] = label_props["centroid_y"].values * DNB_gap
        data = pd.merge(data, label_props, on=["cell"], how="inner")

    if cellbin is True:
        data = data[["geneID", "centroid_x", "centroid_y", "z", "cell", "MIDCounts"]].groupby(["geneID", "centroid_x", "centroid_y", "z", "cell"])["MIDCounts"].sum().to_frame("MIDCounts").reset_index()
        data.columns = ["geneID", "x", "y", "z", "cell", "MIDCounts"]
        data["obs_index"] = data["cell"]
    else:
        data["obs_index"] = data["x"].astype(str) + "_" + data["y"].astype(str)

    uniq_cell, uniq_gene = data["obs_index"].unique().tolist(), data["geneID"].unique().tolist()

    cell_dict = dict(zip(uniq_cell, range(0, len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(0, len(uniq_gene))))

    data["csr_x_ind"] = data["obs_index"].map(cell_dict)
    data["csr_y_ind"] = data["geneID"].map(gene_dict)

    # X
    X = csr_matrix((data["MIDCounts"], (data["csr_x_ind"], data["csr_y_ind"])), shape=(len(uniq_cell), len(uniq_gene)))

    # obs
    del data["geneID"], data["MIDCounts"], data["csr_x_ind"], data["csr_y_ind"]
    if z is None:
        del data["z"]
    obs = data.drop_duplicates()
    obs["slice"] = slice
    obs.set_index("obs_index", inplace=True)

    # var
    var = pd.DataFrame({"gene_short_name": uniq_gene})
    var.set_index("gene_short_name", inplace=True)

    # obsm
    coords = obs[["x", "y"]].values
    obsm = {"spatial": coords}

    adata = AnnData(X=X, obs=obs, var=var, obsm=obsm)

    return adata