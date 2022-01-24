from typing import Union

from ..io import read_lasso
from ..obtain_dataset import fm_gene2GO


def find_nuclear_genes(
        file: str = None,
        save: str = None,
        gene_num: Union[str, int] = "all"
):
    """
    Finding nuclear localized genes in slices based on GO annotations.

    Parameters
    ----------
    file: `str`  (default: `None`)
        Lasso file.
    save: `str` (default: `None`)
        Output filename.
    gene_num: `str` or `list` (default: `'all'`)
        The number of nuclear localized genes. If gene_num is `'all'`, output all nuclear localized genes found.
    Returns
    -------
    new_lasso: `pd.DataFrame`

    """

    # load data
    lasso = read_lasso(file=file)
    lasso_genes = lasso["geneID"].unique().tolist()

    # the GO terms for a particular gene list
    go_data = fm_gene2GO(gene=lasso_genes, gene_identifier="symbol", GO_namespace="cellular_component")
    # go_data.to_excel("E14-16h_a_S09_cellular_component.xlsx", index=False)

    # find nuclear-localized genes
    nucleus_info = "chromosome|chromatin|euchromatin|heterochromatin|nuclear|nucleus|nucleoplasm|nucleolus|transcription factor"
    nucleus_data = go_data[go_data["GO name"].str.contains(nucleus_info)]
    nucleus_data = nucleus_data[~ nucleus_data["GO name"].str.contains("mitochond")]
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
        genes_exp = new_lasso[["geneID", "MIDCounts"]].groupby(["geneID"])["MIDCounts"].sum().to_frame(
            "MIDCounts").reset_index()
        genes_exp.sort_values(by=["MIDCounts", "geneID"], inplace=True, ascending=False)
        top_num_genes = genes_exp["geneID"].head(gene_num)
        new_lasso = new_lasso[new_lasso["geneID"].isin(top_num_genes)]
    print(f"The number of nuclear localized genes found is: {len(new_lasso['gennID'].unique())}.")

    # save
    if save is not None:
        new_lasso.to_csv(save, sep="\t", index=False)

    return new_lasso
