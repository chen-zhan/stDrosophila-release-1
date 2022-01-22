# Annotate a gene list based on the flyaltas2 database
import pandas as pd
from .anno import symbol2fbgn, gene2tissue
from typing import Union, Optional


def genes_flyaltas2(
        genes: Union[str, list] = None,
        gene_nametype: Optional[str] = "symbol",
        stage: Optional[str] = 'male_adult',
        enrich_threshold: Optional[float] = 1.0,
        fbgn_path: Optional[str] = 'deml_fbgn.tsv.gz'
) -> pd.DataFrame:
    """
    Annotate a gene list based on the flyaltas2 database

    Parameters
    ----------
    genes: `str` or `list` (default: `None`)
        The name of a gene, or a list of genes.
    gene_nametype : `str` (default: `'symbol'`)
        Type of gene name, including `'symbol'` and `'FBgn'`.
    stage: `str` (default: `'male_adult'`)
        The developmental stages of Drosophila melanogaster, including `larval`, `female_adult` and `male_adult`.
    enrich_threshold: `float` (default: `1.0`)
        Threshold for filtering enrichment in FlyAtlas 2.
    fbgn_path: `str` (default: `'deml_fbgn.tsv.gz'`)
        Absolute path to the deml_fbgn.tsv.gz.

    Returns
    -------
    anno_genes: `pandas.DataFrame`
        The genes and the particular tissues in which the genes are specifically expressed of each group.

    """
    genes = [genes] if isinstance(genes, str) else genes
    fbgn_names = symbol2fbgn(gene=genes, datapath=fbgn_path) if gene_nametype is "symbol" else genes

    # Find the particular tissue in which the gene is specifically expressed
    anno_genes = pd.DataFrame()
    for fbgn_name in fbgn_names:
        particular_tissues = gene2tissue(fbgn_name, stage, enrich_threshold)
        if particular_tissues is not None:
            anno_genes = pd.concat([anno_genes, particular_tissues], axis=0)

    return anno_genes.astype(str)
