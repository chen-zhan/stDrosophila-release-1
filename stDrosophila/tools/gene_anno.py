import gzip
import io
import os
import logging as logg
import re
import time

import numpy as np
import pandas as pd

from anndata import AnnData
from pandas import DataFrame
from typing import Optional, Union, Sequence



def genes_flyaltas2(
    genes: Union[str, list] = None,
    gene_nametype: Optional[str] = "symbol",
    stage: Optional[str] = "male_adult",
    enrich_threshold: Optional[float] = 1.0,
    fbgn_path: Optional[str] = "deml_fbgn.tsv.gz",
    Top50_path: Optional[str] = "Top50_path",
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
        The developmental stages of Drosophila melanogaster. Available stages are:
            * `'larval'`
            * `'female_adult'`
            * `'male_adult'`
    enrich_threshold: `float` (default: `1.0`)
        Threshold for filtering enrichment in FlyAtlas 2.
    fbgn_path: `str` (default: `'deml_fbgn.tsv.gz'`)
        Absolute path to the deml_fbgn.tsv.gz.
    Top50_path: `str` 
        Absolute path to the Top50 special gene of tissue from 
    Returns
    -------
    anno_genes: `pandas.DataFrame`
        The genes and the particular tissues in which the genes are specifically expressed of each group.
    """
    genes = [genes] if isinstance(genes, str) else genes
    fbgn_names = (
        symbol2fbgn(gene=genes, datapath=fbgn_path)
        if gene_nametype is "symbol"
        else genes
    )

    # Find the particular tissue in which the gene is specifically expressed
    anno_genes = pd.DataFrame()
    for fbgn_name in fbgn_names:
        particular_tissues = gene2tissue(fbgn_name, stage, enrich_threshold, Top50_path)
        if particular_tissues is not None:
            anno_genes = pd.concat([anno_genes, particular_tissues], axis=0)

    return anno_genes.astype(str)




def symbol2fbgn(gene: Union[str, list] = None, datapath: Optional[str] = None):

    FBgn_datasets = pd.read_csv(datapath, sep="\t")

    if isinstance(gene, str):
        FBgn_data = FBgn_datasets[FBgn_datasets["Symbol"] == gene]
        if len(FBgn_data.index) == 0:
            FBgn_data = FBgn_datasets[
                FBgn_datasets["Symbol_synonym(s)"].map(
                    lambda s: gene in str(s).strip().split("|")
                )
            ]
        return FBgn_data["FBgn"].iloc[0]
    elif isinstance(gene, list):
        fbgn_list = []
        for onegene in gene:
            FBgn_data = FBgn_datasets[FBgn_datasets["Symbol"] == onegene]
            if len(FBgn_data.index) == 0:
                FBgn_data = FBgn_datasets[
                    FBgn_datasets["Symbol_synonym(s)"].map(
                        lambda s: onegene in str(s).strip().split("|")
                    )
                ]
            fbgn_list.append(FBgn_data["FBgn"].iloc[0])
        return fbgn_list
    

def gene2tissue(gene, stage, enrich_threshold, Top50_path):
    # obtain "gene_info" and "tissue_info" from Top50_path
    top50_genes=os.listdir(Top50_path)
    if gene not in top50_genes:
        print ("gene no exits in Top50:"+gene)
    else:
        gene_info_dir = os.path.abspath(Top50_path+"/"+gene+"/gene.csv")
        tissue_info_dir = os.path.abspath(Top50_path+"/"+gene+"/tissue.csv")
        gene_info=pd.read_csv(gene_info_dir, index_col= 0) if os.path.exists(gene_info_dir) else None
        tissue_info=pd.read_csv(tissue_info_dir, index_col= 0) if os.path.exists(tissue_info_dir) else None
        
    enrichment_stage = f"enrichment_{stage}"

    # filter stage
    if gene_info is not None and tissue_info is not None:
        gene_tissue = gene_info
        if tissue_info.loc["Whole body", enrichment_stage] == 1:
            tissue_info.drop(index="Whole body", inplace=True)
            tissue_info = tissue_info[tissue_info[enrichment_stage] != 0]
            Percentile = np.percentile(
                tissue_info[enrichment_stage], [0, 25, 50, 75, 100]
            )
            IQR = Percentile[3] - Percentile[1]
            UpLimit = Percentile[3] + IQR * 1.5
            filter_data = tissue_info[tissue_info[enrichment_stage] >= UpLimit]
            filter_data = filter_data[filter_data[enrichment_stage] >= enrich_threshold]
            filter_data["tissue"] = (
                filter_data.index.astype(str)
                + "("
                + filter_data[enrichment_stage].astype(str)
                + ")"
            )
            filter_tissues = filter_data["tissue"].tolist()
            if 1 <= len(filter_tissues):
                gene_tissue["enrichment"] = str(filter_tissues)
                return gene_tissue
            else:
                logg.warning(f"No particular tissue for {gene}.")
                return None
        else:
            logg.warning(f"No particular tissue for {gene}.")
            return None
        
###test###
genes_flyaltas2(genes=["Act88F","CecC"],gene_nametype="symbol", stage="male_adult", 
                fbgn_path= 'd:\\BGI\\fly\\tissue marker\\deml_fbgn.tsv.gz', Top50_path= 'd:\\BGI\\fly\\tissue marker\\Top50_Data')
