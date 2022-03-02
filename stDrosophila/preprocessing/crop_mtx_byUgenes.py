import io
import gzip
import pandas as pd
import numpy as np
from skimage import morphology
import skimage.filters as filters
from sklearn.covariance import EllipticEnvelope
from skimage.morphology import disk


def _convert_binsize(bin1_data, binkt=None, binout=None):
    bin1_data["geneID"] = (
        bin1_data["geneID"].map(lambda g: str(g).strip('"')).astype(str)
    )

    # binkt
    binkt_data = bin1_data.copy()
    binkt_data["x"] = (binkt_data["x"] / binkt).astype(int) * binkt
    binkt_data["y"] = (binkt_data["y"] / binkt).astype(int) * binkt
    binkt_data["binkt_coords"] = (
        binkt_data["x"].astype(str) + "_" + binkt_data["y"].astype(str)
    )
    # binout
    binout_data = bin1_data.copy()
    binout_data["x"] = (binout_data["x"] / binout).astype(int) * binout
    binout_data["y"] = (binout_data["y"] / binout).astype(int) * binout
    binout_data["binkt_coords"] = binkt_data["binkt_coords"]
    binout_data = binout_data.reindex(
        columns=["geneID", "x", "y", "MIDCounts", "binkt_coords"]
    )
    # merge
    binkt_data = (
        binkt_data.groupby(["x", "y", "binkt_coords", "geneID"])["MIDCounts"]
        .sum()
        .to_frame("MIDCounts")
        .reset_index()
    )
    binkt_data = binkt_data.reindex(
        columns=["geneID", "x", "y", "MIDCounts", "binkt_coords"]
    )
    binout_data = (
        binout_data.groupby(["x", "y", "binkt_coords", "geneID"])["MIDCounts"]
        .sum()
        .to_frame("MIDCounts")
        .reset_index()
    )
    binout_data = binout_data.reindex(
        columns=["geneID", "x", "y", "MIDCounts", "binkt_coords"]
    )
    return binkt_data, binout_data


def _read_datasets():
    with gzip.open("../data/bdgp_flyfish.tsv.gz", "rb") as input_file:
        with io.TextIOWrapper(input_file, encoding="utf-8") as dec:
            dec_list = [str(i).strip("\n").split(",") for i in dec.readlines()]
            dec_data = pd.DataFrame(dec_list[1:], columns=dec_list[0], dtype=str)
    return dec_data


def filterUGenes(stage=None, ugfamily=None):
    avail_stages = {
        "stage 1-3 (BDGP in situ)",
        "stage 4-6 (BDGP in situ)",
        "stage 7-8 (BDGP in situ)",
        "stage 9-10 (BDGP in situ)",
        "stage 11-12 (BDGP in situ)",
        "stage 13-16 (BDGP in situ)",
        "stage 1-3 (fly-FISH)",
        "stage 4-5 (fly-FISH)",
        "stage 6-7 (fly-FISH)",
        "stage 8-9 (fly-FISH)",
    }

    stages = [stage] if isinstance(stage, str) else stage
    ugfamilies = [ugfamily] if isinstance(ugfamily, str) else ugfamily

    for s in stages:
        if s not in avail_stages:
            raise ValueError(
                f"The developmental stage of Drosophila melanogaster must be one of {avail_stages}."
            )

    flydata = _read_datasets()
    stages_flydata = flydata[flydata["StageRange"].isin(stages)]
    if ugfamily is None:
        ubiquitous_genes = (
            stages_flydata[
                stages_flydata["mRNAExpressionTerms"].str.contains("ubiquitous")
            ]["Gene_Symbol"]
            .unique()
            .tolist()
        )
    else:
        ubiquitous_genes = (
            stages_flydata[
                stages_flydata["mRNAExpressionTerms"].str.contains("ubiquitous")
                & stages_flydata["Gene_Symbol"].isin(ugfamilies)
            ]["Gene_Symbol"]
            .unique()
            .tolist()
        )
    return ubiquitous_genes


def crop_midimg(data, bin=None, otsu=6, opening=2, contamination=0.05, med=3):
    def otsu_opening(img_gray, nbins=6, open=2):
        """OTSU threshold segmentation image and open operation filter processing image"""
        theta = filters.threshold_otsu(img_gray, nbins=nbins)
        img_seg = np.zeros(img_gray.shape)
        img_seg[img_gray > theta] = 1
        img_d = morphology.opening(img_seg, morphology.square(open))
        # img_d=morphology.closing(img_seg,morphology.square(2))
        return img_d

    def anomaly_detection(coords, contamination=0.05):
        """Outlier handling"""
        model = EllipticEnvelope(contamination=contamination)
        model.fit(coords)
        y_predict = model.predict(coords)
        coords["y_predict"] = y_predict
        ad_coords = coords[coords["y_predict"] != -1]
        ad_coords.drop(columns="y_predict", axis=1, inplace=True)
        return ad_coords

    def fill_gap(coords, bin):
        """Fill in the gaps in the image"""
        fill_coords_list1 = []
        for name1, group1 in coords.groupby("x"):
            y_min, y_max = (
                group1["y"].astype(float).min(),
                group1["y"].astype(float).max(),
            )
            fill_coords_list1.extend(
                [[name1, i] for i in range(y_min, y_max + bin, bin)]
            )
        fill_coords_data1 = pd.DataFrame(fill_coords_list1, columns=["x", "y"])
        fill_coords_data1 = pd.concat([coords, fill_coords_data1], axis=0).astype(float)
        fill_coords_data1.drop_duplicates(inplace=True)
        fill_coords_list2 = []
        for name2, group2 in fill_coords_data1.groupby("y"):
            x_min, x_max = (
                group2["x"].astype(float).min(),
                group2["x"].astype(float).max(),
            )
            fill_coords_list2.extend(
                [[i, name2] for i in range(x_min, x_max + bin, bin)]
            )
        return pd.DataFrame(fill_coords_list2, columns=["x", "y"], dtype=float)

    def edge_smoothing(coords, med):
        """Median filter to achieve image edge smoothing"""
        coords["value"] = 1
        coords_table = pd.pivot_table(
            coords,
            index=["y"],
            columns=["x"],
            values="value",
            aggfunc=np.sum,
            fill_value=0,
        ).astype("float64")
        smooth_mtx = filters.median(coords_table.values, disk(med))
        smooth_table = pd.DataFrame(
            smooth_mtx, columns=coords_table.columns, index=coords_table.index
        )
        smooth_table["y"] = smooth_table.index
        smooth_coords = pd.melt(smooth_table, id_vars=["y"])
        smooth_coords = smooth_coords[smooth_coords["value"] != 0]
        return smooth_coords

    img = otsu_opening(img_gray=data.values, nbins=otsu, open=opening)
    img_table = pd.DataFrame(img, columns=data.columns, index=data.index)
    img_table["y"] = img_table.index
    img_data = pd.melt(img_table, id_vars=["y"])
    coords = img_data[img_data["value"] != 0][["x", "y"]]
    ad_coords = anomaly_detection(coords=coords, contamination=contamination)
    fill_coords = fill_gap(coords=ad_coords, bin=bin)
    smooth_coords = edge_smoothing(coords=fill_coords, med=med)
    reslut_coords = [
        f"{int(smooth_coords['x'].loc[i])}_{int(smooth_coords['y'].loc[i])}"
        for i in smooth_coords.index
    ]
    return reslut_coords


def cropbyUG(
    bin1_data,
    stage=None,
    ugfamily=None,
    binkt=5,
    binout=20,
    otsu=6,
    opening=2,
    contamination=0.05,
    med=3,
):
    """
    Cropping based on ubiquitous genes' expression.

    Parameters
    ----------
    bin1_data:
        bin1 lasso data.
    stage: `str` (default: `None`)
        The developmental stage of Drosophila melanogaster, including:
        `stage 1-3 (BDGP in situ)`, `stage 4-6 (BDGP in situ)`, `stage 7-8 (BDGP in situ)`,
        `stage 9-10 (BDGP in situ)`, `stage 11-12 (BDGP in situ)`, `stage 13-16 (BDGP in situ)`,
        `stage 1-3 (fly-FISH)`, `stage 4-5 (fly-FISH)`, `stage 6-7 (fly-FISH)`, `stage 8-9 (fly-FISH)`.
    ugfamily: `str` (default: `None`)
        ubiquitous gene family. e.g., ugfamily=`Rps`.
    binkt: `int` (default: `5`)
        The size of the bin of the data to be processed in the middle.
    binout: `int` (default: `20`)
        The size of the bin of the final output data.

    Examples
    --------
    >> bin1_data = pd.read_csv(r"bin1_lasso.txt", sep="\t")
    >> binout_data = cropbyUG(bin1_data, stage="stage 13-16 (BDGP in situ)", ugfamily="Rps", binkt=5, binout=20)
    """
    binkt_data, binout_data = _convert_binsize(
        bin1_data=bin1_data, binkt=binkt, binout=binout
    )

    ugenes = filterUGenes(stage=stage, ugfamily=ugfamily)
    binkt_data = binkt_data[binkt_data["geneID"].isin(ugenes)]

    binkt_data["MIDCounts"] = binkt_data["MIDCounts"] / binkt_data["MIDCounts"].max()
    binkt_table = pd.pivot_table(
        binkt_data,
        index=["y"],
        columns=["x"],
        values="MIDCounts",
        aggfunc=np.sum,
        fill_value=0,
    ).astype("float64")
    new_coords = crop_midimg(
        data=binkt_table,
        bin=binkt,
        otsu=otsu,
        opening=opening,
        contamination=contamination,
        med=med,
    )

    binout_data = binout_data[binout_data["binkt_coords"].isin(new_coords)]
    binout_data.drop(columns="binkt_coord", axis=1)
    binout_data = (
        binout_data.groupby(["x", "y", "geneID"])["MIDCounts"]
        .sum()
        .to_frame("MIDCounts")
        .reset_index()
    )
    return binout_data
