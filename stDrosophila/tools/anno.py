import gzip
import io
import logging as logg
import re
import requests
import time

import numpy as np
import pandas as pd

from anndata import AnnData
from pandas import DataFrame
from typing import Optional, Union, Sequence

from ..obtain_dataset import flyatlas2


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


def get_genesummary(
    data: DataFrame,
    geneby: Optional[str] = None,
    gene_nametype: Optional[str] = "symbol",
    datapath: Optional[str] = None,
):
    """Get gene's summary from FlyBase."""
    avail_gene_nametypes = {"symbol", "FBgn"}
    if gene_nametype not in avail_gene_nametypes:
        raise ValueError(f"Type of gene name must be one of {avail_gene_nametypes}.")

    gene_names = data[geneby].tolist()
    if gene_nametype is "symbol":
        gene_names = [symbol2fbgn(gene_name) for gene_name in gene_names]

    with gzip.open(datapath, "rb") as input_file:
        with io.TextIOWrapper(input_file, encoding="utf-8") as dec:
            dec_list = [
                str(i).strip().split("\t")
                for i in dec.readlines()
                if i.startswith("#") is False
            ]
    summaries = pd.DataFrame(dec_list, columns=["FlyBase ID", "gene_summary"])

    summary_dict = {}
    for gene_name in gene_names:
        summary_dict[gene_name] = summaries[summaries["FlyBase ID"] == gene_name][
            "gene_summary"
        ].iloc[0]

    data["gene_summary"] = data[geneby].map(lambda g: summary_dict[g])

    return data


def gene2tissue(gene, stage, enrich_threshold):
    # obtain data from flyatlas 2
    gene_info, tissue_info = flyatlas2(fbgn=gene, stage=stage)
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


def anno_flyatlas2(
    adata: AnnData,
    stage: Optional[str] = "male_adult",
    gene_nametype: Optional[str] = "symbol",
    groupby: Optional[str] = None,
    groups: Union[str, Sequence[str]] = None,
    n_genes: Optional[int] = 20,
    enrich_threshold: Optional[float] = 1.0,
    add_genesummary: bool = True,
    obs_key: Optional[str] = "auto_anno",
    key: Optional[str] = "rank_genes_groups",
    key_added: Optional[str] = "marker_genes_anno",
    copy: bool = False,
    gene_summaries_path: Optional[str] = "automated_gene_summaries.tsv.gz",
    fbgn_path: Optional[str] = "deml_fbgn.tsv.gz",
):
    """\
    Annotate clustered groups based on FlyAtlas2.

    Parameters
    ----------
    adata: :class:`~anndata.AnnData`
        A clustered AnnData.
    stage: `str` (default: `'male_adult'`)
        The developmental stages of Drosophila melanogaster, including `'larval'`, `'female_adult'` and `'male_adult'`.
    gene_nametype : `str` (default: `'symbol'`)
        Type of gene name, including `'symbol'` and `'FBgn'`.
    groupby: `str` (default: `None`)
        The key of the observation grouping to consider.
    groups: `str` (default: `None`)
        The groups for which to be annotated.
    n_genes: `int` (default: `20`)
        The number of genes that annotate groups.
    enrich_threshold: `float` (default: `1.0`)
        Threshold for filtering enrichment in FlyAtlas 2.
    add_genesummary: `bool` (default: `False`)
        Whether to add summaries of all genes to the output data.
    obs_key: `str` (default: `'auto_anno'`)
        The key in `adata.obs` information is saved to.
    key: `str` (default: `'rank_genes_groups'`)
        The key of the marker gene stored in the `adata.uns` information.
    key_added: `str` (default: `'marker_genes_anno'`)
        The key in `adata.uns` information is saved to.
    copy: `bool` (default: `False`)
        Whether to copy adata or modify it inplace.
    gene_summaries_path: `str` (default: `'automated_gene_summaries.tsv.gz'`)
        Absolute path to the automated_gene_summaries.tsv.gz.
    fbgn_path: `str` (default: `deml_fbgn.tsv.gz'`)
        Absolute path to the deml_fbgn.tsv.gz.

    Returns
    -------
    adata: `AnnData`
            :attr:`~anndata.AnnData.uns`\\ `['marker_genes_anno']`
                The genes and the particular tissues in which the genes are specifically expressed of each group.
            :attr:`~anndata.AnnData.obs`\\ `['auto_anno']`
                Automated annotation results.
    anno_data: `pandas.DataFrame`
            The genes and the particular tissues in which the genes are specifically expressed of each group.

    Notes
    -----
    The annotation data come from FlyAtlas 2. For more information about FlyAtlas 2, please see <flyatlas2.org>.
    """
    avail_stages = {"larval", "female_adult", "male_adult"}
    if stage not in avail_stages:
        raise ValueError(
            f"The developmental stage of Drosophila melanogaster method must be one of {avail_stages}."
        )

    avail_gene_nametypes = {"symbol", "FBgn"}
    if gene_nametype not in avail_gene_nametypes:
        raise ValueError(f"Type of gene name must be one of {avail_gene_nametypes}.")

    adata = adata.copy() if copy else adata
    adata._sanitize()
    # The groups for which to be annotated
    group_names = adata.uns[key]["names"].dtype.names if groups is None else groups
    key_added = "marker_genes_anno" if key_added is None else key_added
    adata.uns[key_added] = {}
    adata.obs[obs_key] = adata.obs[groupby].astype(str)
    anno_data = pd.DataFrame()

    for group_name in group_names:
        # The genes used to annotate groups
        gene_names = list(adata.uns[key]["names"][group_name][:n_genes])
        fbgn_names = (
            symbol2fbgn(gene=gene_names, datapath=fbgn_path)
            if gene_nametype is "symbol"
            else gene_names
        )
        # Find the particular tissue in which the gene is specifically expressed
        print(f"----- Start group {group_name}, including {gene_names}")
        anno_genes = pd.DataFrame()
        for fbgn_name in fbgn_names:
            particular_tissues = gene2tissue(fbgn_name, stage, enrich_threshold)
            if particular_tissues is not None:
                anno_genes = pd.concat([anno_genes, particular_tissues], axis=0)
                time.sleep(secs=30)
        group_name = str(group_name)
        anno_genes = anno_genes.astype(str)
        # Add gene annotation information to adata.uns
        adata.uns[key_added][group_name] = anno_genes
        # Add gene annotation information to data
        anno_genes["cluster"] = group_name
        anno_data = pd.concat([anno_data, anno_genes], axis=0)
        tissues = [
            re.sub(r"\(\d*\.\d*\)", "", str(i).strip("[]").split(",")[0].strip("'"))
            for i in anno_genes["enrichment"].tolist()
        ]
        # Add gene annotation information to adata.obs
        n_tissues = {tissue: tissues.count(tissue) for tissue in tissues}
        group_tissues = [
            key for key in n_tissues.keys() if n_tissues[key] == max(n_tissues.values())
        ]
        group_anno = ""
        for i in range(len(group_tissues)):
            group_anno = (
                group_anno + group_tissues[i]
                if i == 0
                else group_anno + "+" + group_tissues[i]
            )
        adata.obs[obs_key] = adata.obs[obs_key].map(
            lambda c: group_anno if c == group_name else c
        )

    if add_genesummary:
        # Add summaries of all genes to the output data.
        anno_data = get_genesummary(
            data=anno_data,
            geneby="FlyBase ID",
            gene_nametype="FBgn",
            datapath=gene_summaries_path,
        )

    return adata, anno_data if copy else None, anno_data
