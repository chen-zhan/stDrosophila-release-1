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
            "cell": "category",
        })

    lasso_data["geneID"] = lasso_data.geneID.astype(str).str.strip('"')

    return lasso_data


def lasso2adata(
    data: pd.DataFrame,
    slice: Optional[str] = None,
    label_path: Optional[str] = None,
    DNB_gap: Optional[float] = 0.5,
    z: Union[float] = None,
    z_gap: Union[float] = None
) -> AnnData:
    """A helper function that facilitates constructing an AnnData object suitable for downstream spateo analysis

    Parameters
    ----------
        data: `pandas.DataFrame`
            A string that points to the directory and filename of spatial transcriptomics dataset, produced by the
            stereo-seq method from BGI.
        slice: `str` or None (default: None)
            Name of the slice. Will be used when displaying multiple slices.
        label_path: `str` or None (default: None)
            A string that points to the directory and filename of cell segmentation label matrix(Format:`.npy`).
            If not None, the results of cell segmentation will be used, and param `binsize` will be ignored.
        DNB_gap: `float` (default: `0.5`)

        z: `float` (default: None)

        z_gap: `float` (default: None)


    Returns
    -------
        adata: :class:`~anndata.AnnData`
            An AnnData object. Each row of the AnnData object correspond to a spot (aggregated with multiple bins). The
            `spatial` key in the .obsm corresponds to the x, y coordinates of the centroids of all spot.
    """
    # physical coords
    data["x"] = (data["x"].values - data["x"].values.min()) * DNB_gap
    data["y"] = (data["y"].values - data["y"].values.min()) * DNB_gap
    if z is not None:
        data["z"] = z * z_gap / DNB_gap

    # get cell name
    if label_path is not None:
        # TODO: Get cell names using labels
        if label_path.endswith(".gz"):
            with gzip.open(label_path, "r") as f:
                label_mtx = np.load(f)
        else:
            label_mtx = np.load(label_path)

        props = measure.regionprops_table(label_mtx, properties=("label", "area", "centroid"))
        label_props = pd.DataFrame(props)
        label_props.columns = ["cell", "area", "x", "y"]
        label_props["cell"] = label_props["cell"].astype("category")

        del data["x"], data["y"]
        data = pd.merge(data, label_props, on=["cell"], how="inner")
        data = data[
            ["geneID", "x", "y", "cell", "area", "MIDCounts"]
        ].groupby(["geneID", "x", "y", "cell", "area"])["MIDCounts"].sum().to_frame("MIDCounts").reset_index()
    else:
        data["cell"] = data["x"].astype(str) + "_" + data["y"].astype(str)

    uniq_cell, uniq_gene = data["cell"].unique().tolist(), data["geneID"].unique().tolist()

    cell_dict = dict(zip(uniq_cell, range(0, len(uniq_cell))))
    gene_dict = dict(zip(uniq_gene, range(0, len(uniq_gene))))

    data["csr_x_ind"] = data["cell"].map(cell_dict)
    data["csr_y_ind"] = data["geneID"].map(gene_dict)

    X = csr_matrix(
        (data["MIDCounts"], (data["csr_x_ind"], data["csr_y_ind"])),
        shape=(len(uniq_cell), len(uniq_gene)),
    )

    # obs
    obs = data[["x", "y", "cell"]].drop_duplicates()
    obs["slice"] = slice
    obs.set_index("cell", inplace=True)

    # var
    var = pd.DataFrame({"gene_short_name": uniq_gene})
    var.set_index("gene_short_name", inplace=True)

    # obsm
    coords = obs[["x", "y"]].values
    obsm = {"spatial": coords}

    adata = AnnData(X=X, obs=obs, var=var, obsm=obsm)

    return adata
