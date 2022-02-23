import os
import pandas as pd


def bin1tobinx(bin1_data, binx=None, save=None):
    """
    Converts bin size from 1 to other bin size.

    Parameters
    ----------
    bin1_data:
        bin1 lasso data.
    binx: `int` (default: `None`)
        The size of bin after conversion.
    save: `str` (default: `None`)
        If save is not None, save is the path to save the lasso data.

    Examples
    --------
    >> bin1_data = pd.read_csv(r"bin1_lasso.txt", sep="\t")
    >> bin5_data = bin1tobinx(bin1_data, binx=5, save="bin5_lasso.gem.gz")
    """

    bin1_data["x"] = (bin1_data["x"] / binx).astype(int) * binx
    bin1_data["y"] = (bin1_data["y"] / binx).astype(int) * binx
    binx_data = (
        bin1_data.groupby(["x", "y", "geneID"])["MIDCounts"]
            .sum()
            .to_frame("MIDCounts")
            .reset_index()
    )
    binx_data = binx_data.reindex(columns=["geneID", "x", "y", "MIDCounts"])
    if save is not None:
        binx_data.to_csv(save, index=False, sep="\t")
    return binx_data


def binxtobiny(binx_data, bin1_data, binx=None, biny=1, save=None):
    """
    Converts bin size from x to 1.

    Parameters
    ----------
    binx_data:
        binx lasso data.
    bin1_data：
        bin1 lasso data.
    binx: `int` (default: `None`)
        The size of bin before conversion.
    biny: `int` (default: `None`)
        The size of bin after conversion.
    save: `str` (default: `None`)
        If save is not None, save is the path to save the lasso data.

    Examples
    --------
    >> bin1_data = pd.read_csv(r"bin1_lasso.txt", sep="\t")
    >> bin5_data = bin1tobinx(bin1_data, binx=5, save="bin5_lasso.gem.gz")
    """

    binx_coords = binx_data.loc[:, ["x", "y"]].drop_duplicates()
    binx_coords.index, binx_coords.columns = range(len(binx_coords.index)), ["binx_x", "binx_y"]

    bin1_data["binx_x"] = (bin1_data["x"] / binx).astype(int) * binx
    bin1_data["binx_y"] = (bin1_data["y"] / binx).astype(int) * binx
    bin1_need = pd.merge(bin1_data, binx_coords, on=["binx_x", "binx_y"], how="inner")
    del bin1_need["binx_x"], bin1_need["binx_y"]

    if biny != 1:
        biny_data = bin1tobinx(bin1_need, binx=biny, save=save)
        return biny_data
    else:
        if save is not None:
            bin1_need.to_csv(save, index=False, sep="\t")
        return bin1_need



"/media/yao/Elements SE/BGI_Paper/mouse_brain/stereo/day3_brain_0110/1_Crop/crop_bin1"