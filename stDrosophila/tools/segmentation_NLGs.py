import gzip

import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix
from typing import Optional, Union

from ..io import read_lasso
from ..obtain_dataset import fm_gene2GO


def find_nuclear_genes(
    path: str = None, save: str = None, gene_num: Union[str, int] = "all"
) -> pd.DataFrame:
    """
    Finding nuclear localized genes in slices based on GO annotations.

    Parameters
    ----------
    path: `str`  (default: `None`)
        Path to lasso file.
    save: `str` (default: `None`)
        Output filename.
    gene_num: `str` or `list` (default: `'all'`)
        The number of nuclear localized genes. If gene_num is `'all'`, output all nuclear localized genes found.

    Returns
    -------
    new_lasso: `pd.DataFrame`

    """

    # load data
    lasso = read_lasso(path=path)
    lasso_genes = lasso["geneID"].unique().tolist()

    # the GO terms for a particular gene list
    go_data = fm_gene2GO(
        gene=lasso_genes, gene_identifier="symbol", GO_namespace="cellular_component"
    )
    # go_data.to_excel("E14-16h_a_S09_cellular_component.xlsx", index=False)

    # find nuclear-localized genes
    nucleus_info = "chromosome|chromatin|euchromatin|heterochromatin|nuclear|nucleus|nucleoplasm|nucleolus|transcription factor"
    nucleus_data = go_data[go_data["GO name"].str.contains(nucleus_info)]
    nucleus_data = nucleus_data[~nucleus_data["GO name"].str.contains("mitochond")]
    nucleus_genes = nucleus_data["gene symbol"].unique().tolist()

    # remove pseudo positives
    nucleus_filter_data = go_data[go_data["gene symbol"].isin(nucleus_genes)]
    nucleus_filter_groups = nucleus_filter_data.groupby(["gene symbol"])["GO name"]

    nucleus_filter_genes = []
    for i, group in nucleus_filter_groups:
        tf = group.str.contains(nucleus_info).unique().tolist()
        if len(tf) == 1 and tf[0] is True:
            nucleus_filter_genes.append(i)
    new_lasso = lasso[lasso["geneID"].isin(nucleus_filter_genes)]

    # determine the final number of genes obtained
    if gene_num is not "all":
        genes_exp = (
            new_lasso[["geneID", "MIDCounts"]]
            .groupby(["geneID"])["MIDCounts"]
            .sum()
            .to_frame("MIDCounts")
            .reset_index()
        )
        genes_exp.sort_values(by=["MIDCounts", "geneID"], inplace=True, ascending=False)
        top_num_genes = genes_exp["geneID"].head(gene_num)
        new_lasso = new_lasso[new_lasso["geneID"].isin(top_num_genes)]
    print(
        f"The number of nuclear localized genes found is: {len(new_lasso['geneID'].unique())}."
    )

    # save
    if save is not None:
        new_lasso.to_csv(save, sep="\t", index=False)

    return new_lasso


def mapping2lasso(
    total_file, nucleus_file, cells_file: str = None, save: Optional[str] = None
) -> pd.DataFrame:
    """
    Map cell type information to the original lasso file.

    Parameters
    ----------
    total_file: `str`  (default: `None`)
        Lasso file containing all genes.
    nucleus_file: `str` (default: `None`)
        Lasso file containing only nuclear localized genes.=
    cells_file: `str` (default: `None`)
        Matrix file generated by cell segmentation.
    save: `str` (default: `None`)
        Path to save the newly generated lasso file.

    Returns
    -------
    total_cells: `pd.DataFrame`
        The columns of the dataframe include 'geneID', 'x', 'y', 'MIDCounts', 'cell'.

    """

    total_lasso = read_lasso(path=total_file)
    nucleus_lasso = read_lasso(path=nucleus_file)

    # cells processing
    if cells_file.endswith(".gz"):
        with gzip.open(cells_file, "r") as f:
            mtx = coo_matrix(np.load(f))
    else:
        mtx = coo_matrix(np.load(cells_file))

    x = pd.Series(mtx.row) + np.min(nucleus_lasso["x"])
    y = pd.Series(mtx.col) + np.min(nucleus_lasso["y"])
    value = pd.Series(mtx.data)
    cells = pd.concat([x, y, value], axis=1)
    cells.columns = ["x", "y", "cell"]

    # map to the total lasso file
    total_cells = pd.merge(total_lasso, cells, on=["x", "y"], how="inner")
    total_cells["cell"] = total_cells["cell"].astype(str)
    # save
    if save is not None:
        total_cells.to_csv(save, sep="\t", index=False)

    return total_cells
