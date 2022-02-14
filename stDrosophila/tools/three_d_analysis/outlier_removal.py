
import anndata as ad
import numpy as np
import pandas as pd

from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import KernelDensity


def om_kde(
    adata: ad.AnnData,
    coordsby: str = "spatial",
    percent: float = 0.2,
    kernel: str = "gaussian",
    bandwidth: float = 1.0
):
    """Outlier detection processing based on kernel density estimation."""

    coords = adata.obsm[coordsby].values
    adata.obs["coords_kde"] = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(coords).score_samples(coords)

    CV = adata.obs["coords_kde"].describe(percentiles=[percent])[f"{int(percent*100)}%"]

    return adata[adata.obs["coords_kde"] > CV, :]


def om_EllipticEnvelope(
    adata: ad.AnnData,
    coordsby: str = "spatial",
    percent: float = 0.05,
):
    """Outlier detection processing based on EllipticEnvelope algorithm."""

    coords = pd.DataFrame(adata.obsm[coordsby])
    adata.obs["outlier"] = EllipticEnvelope(contamination=percent).fit(coords).predict(coords)

    return adata[adata.obs["outlier"] != -1, :]

