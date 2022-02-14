import anndata as ad
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, spmatrix


def sct_stereo(adata, filter_hvgs=True):

    to_dense_array = lambda X: np.array(X.todense()) if isinstance(X, spmatrix) else X

    try:
        import stereo as st
    except ImportError:
        raise ImportError("\nplease install Stereo:\n\n\tpip install stereopy")

    adata.var_names_make_unique()
    adata_df = pd.DataFrame(
        to_dense_array(adata.X), columns=adata.var.index, index=adata.obs.index
    )

    st_adata = st.io.anndata_to_stereo(adata)
    st_adata.tl.sctransform(filter_hvgs=filter_hvgs, var_features_n=3000, inplace=True)
    sct_adata = st.io.stereo_to_anndata(st_adata)

    sct_adata_df = pd.DataFrame(
        to_dense_array(sct_adata.X),
        columns=sct_adata.var.index,
        index=sct_adata.obs.index,
    )
    sct_adata_df.sort_index(axis=1, inplace=True)
    sct_fvgs = sct_adata_df.columns.tolist()
    adata_df = adata_df.loc[:, adata_df.columns.isin(sct_fvgs)]
    adata_df.sort_index(axis=1, inplace=True)
    var_df = pd.DataFrame([], index=sct_adata_df.columns)
    sct_adata_new = ad.AnnData(
        X=csr_matrix(sct_adata_df.values), var=var_df, obs=adata.obs
    )
    sct_adata_new.obsm["spatial"] = adata.obsm["spatial"]

    sct_adata_raw = sct_adata_new.copy()
    sct_adata_raw.X = adata_df.values
    sct_adata_raw.obsm["spatial"] = adata.obsm["raw_spatial"]
    del sct_adata_raw.obs["x"]
    del sct_adata_raw.obs["y"]
    del sct_adata_raw.obs["z"]

    return sct_adata_new, sct_adata_raw


def sct_dynamo(
    adata,
    min_cells=5,
    gmean_eps=1,
    n_genes=2000,
    n_cells=None,
    bin_size=20,
    bw_adjust=3,
    inplace=False,
):

    try:
        from dynamo.external import sctransform
    except ImportError:
        raise ImportError(
            "\nplease install dynamo:\n\n"
            "\tIf dynamo is updated, install it by the following method:\n"
            "\t\tpip install dynamo-release\n\n"
            "\tbefore dynamo is updated, install it by the following method:\n"
            "\t\tgit clone https://github.com/aristoteleo/dynamo-release.git\n"
            "\t\tcd ./stDrosophila-release\n"
            "\t\tpip install ."
        )

    to_csr = lambda X: csr_matrix(X) if not isinstance(X, spmatrix) else X
    adata.X = to_csr(adata.X)

    return sctransform(
        adata,
        min_cells=min_cells,
        gmean_eps=gmean_eps,
        n_genes=n_genes,
        n_cells=n_cells,
        bin_size=bin_size,
        bw_adjust=bw_adjust,
        inplace=inplace,
    )
