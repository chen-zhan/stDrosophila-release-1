import numpy as np

from anndata import AnnData
from dynamo.vectorfield import vector_field_function
from dynamo.vectorfield.scVectorField import SparseVFC


def interpolation_SparseVFC(
    adata, genes=None, grid_num=50, lambda_=0.02, lstsq_method="scipy", **kwargs
) -> AnnData:
    """
    predict missing location’s gene expression and learn a continuous gene expression pattern over space.
    (Not only for 3D coordinate data, but also for 2D coordinate data.)

    Parameters
    ----------
        adata: :class:`~anndata.AnnData`
            AnnData object that contains spatial (numpy.ndarray) in the `obsm` attribute.
        genes: `list` (default None)
            Gene list that needs to interpolate.
        grid_num: 'int' (default 50)
            Number of grid to generate. Default is 50. Must be non-negative.
        lambda_: 'float' (default: 0.02)
            Represents the trade-off between the goodness of data fit and regularization. Larger Lambda_ put more
            weights on regularization.
        lstsq_method: 'str' (default: `scipy`)
           The name of the linear least square solver, can be either 'scipy` or `douin`.
        **kwargs：
            Additional parameters that will be passed to SparseVFC function.

    Returns
    -------
     three_d_adata: `Anndata`
        adata containing the interpolated gene expression matrix.
    """

    # X (ndarray (dimension: n_obs x n_features)) – Original data.
    X = adata.obsm["spatial"]

    # V (ndarray (dimension: n_obs x n_features)) – Velocities of cells in the same order and dimension of X.
    V = adata[:, genes].X - adata[:, genes].X.mean(0)

    # Generate grid
    min_vec, max_vec = (
        X.min(0),
        X.max(0),
    )
    min_vec = min_vec - 0.01 * np.abs(max_vec - min_vec)
    max_vec = max_vec + 0.01 * np.abs(max_vec - min_vec)
    Grid_list = np.meshgrid(
        *[np.linspace(i, j, grid_num) for i, j in zip(min_vec, max_vec)]
    )
    Grid = np.array([i.flatten() for i in Grid_list]).T

    # Get the new adata after interpolation (Not only for 3D coordinate data, but also for 2D coordinate data).
    res = SparseVFC(X, V, Grid, lambda_=lambda_, lstsq_method=lstsq_method, **kwargs)
    three_d_func = lambda x: vector_field_function(x, res)
    three_d_adata = adata.copy()
    three_d_adata[:, genes].X = three_d_func(X) + adata[:, genes].X.mean(0)

    return three_d_adata
