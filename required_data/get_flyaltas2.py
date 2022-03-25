#Use this script to download flyaltas2 data from a networked environment
#Or find in required_data/FlyAltas2.zip, which is about the "male_adult"
#The data is applied to the annotation organization

import logging as logg
import requests
import pandas as pd
import numpy as np
import os
from typing import Tuple:


def flyatlas2(
    fbgn: str = None, stage: str = "male_adult"
) -> Tuple[pd.DataFrame, pd.DataFrame] or Tuple[None, None]:
    """For a particular Drosophila gene, find the pattern of expression in different tissues.
    Parameters
    ----------
    fbgn: `str` (default: `None`)
        Gene name (DB identifier).
    stage: `str` (default: `'male_adult'`)
        The developmental stages of Drosophila melanogaster. Available stages are:
            * `'larval'`
            * `'female_adult'`
            * `'male_adult'`
    Returns
    -------
    gene_info: `pd.DataFrame`
    tissue_info: `pd.DataFrame`
    """

    s = requests.session()
    s.keep_alive = False
    try:
        url = f"https://motif.mvls.gla.ac.uk/FA2Direct/index.html?fbgn={fbgn}&tableOut=gene;"
        flyatlas_content = requests.get(url).content.decode().strip()
    except:
        url = f"https://motif.mvls.gla.ac.uk/FA2Direct/index.html?fbgn={fbgn}&tableOut=mir"
        flyatlas_content = requests.get(url).content.decode().strip()
    flyatlas_content = [
        i.strip() for i in flyatlas_content.split("\n") if i.strip() != ""
    ]
    separation = [i for i in flyatlas_content if str(i).startswith("Tissue")]

    if len(separation) == 1:

        gene_info = [
            i.strip().split("\t")
            for i in flyatlas_content
            if str(i).startswith("FlyBase ID")
            or str(i).startswith("Annotation Symbol")
            or str(i).startswith("Symbol")
        ]

        gene_info = pd.DataFrame(gene_info).stack().unstack(0)
        gene_info.columns = gene_info.iloc[0].tolist()
        gene_info.drop(index=0, inplace=True)

        separation_index = flyatlas_content.index(separation[0])
        raw_tissue_info = [i.split("\t") for i in flyatlas_content][
            separation_index + 1 :
        ]
        raw_tissue_info_col = [
            "tissue",
            "FPKM_male_adult",
            "SD_male_adult",
            "enrichment_male_adult",
            "FPKM_female_adult",
            "SD_female_adult",
            "enrichment_female_adult",
            "M/F_maleVfemale",
            "Pval_maleVfemale",
            "FPKM_larval",
            "SD_larval",
            "enrichment_larval",
        ]
        raw_tissue_info = pd.DataFrame(raw_tissue_info, columns=raw_tissue_info_col)
        raw_tissue_info.index = raw_tissue_info["tissue"].tolist()

        tissue_info = raw_tissue_info[
            [f"FPKM_{stage}", f"SD_{stage}", f"enrichment_{stage}"]
        ]
        tissue_info = tissue_info.applymap(lambda x: "0" if x == "-" else x)
        tissue_info[f"enrichment_{stage}"] = tissue_info[f"enrichment_{stage}"].apply(lambda x: x.replace(",","",) if "," in x else x)
        tissue_info[f"enrichment_{stage}"] = tissue_info[f"enrichment_{stage}"].astype(float)
        tissue_info.sort_values(
            by=[f"enrichment_{stage}"], ascending=False, inplace=True
        )

        return gene_info, tissue_info

    elif len(separation) == 0:

        logg.warning(f"There are no data in FlyAtlas 2 for {fbgn}.")

        return None, None

           
os.chdir(r"D:\BGI\fly\tissue marker\FlyAltas2")  
table = pd.read_csv("d:\\BGI\\fly\\tissue marker\\deml_fbgn.tsv.gz", sep="\t")
for fbgn in table["FBgn"]:
        gene_info, tissue_info = flyatlas2(fbgn=fbgn, stage="male_adult")
        if gene_info is not None and tissue_info is not None:
            if (os.path.exists(fbgn)):
                pass
            else:
                print(table[table["FBgn"]==fbgn].index)
                os.mkdir(fbgn)
                gene_info.to_csv(os.path.abspath(fbgn+"\\gene.csv"))
                tissue_info.to_csv(os.path.abspath(fbgn+"\\tissue.csv"))