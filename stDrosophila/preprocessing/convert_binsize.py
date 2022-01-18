import os
import pandas as pd


def bin1tobinx(bin1_data, binx=None, save=None, save_format="gzip"):
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
    save_format: `str` (default: `'gzip'`)
        Saved file format.

    Examples
    --------
    >> bin1_data = pd.read_csv(r"bin1_lasso.txt", sep="\t")
    >> bin5_data = bin1tobinx(bin1_data, binx=5, save="bin5_lasso.txt")
    """

    bin1_data['geneID'] = bin1_data['geneID'].map(lambda g: str(g).strip('"')).astype(str)
    binx_data = bin1_data.copy()
    binx_data['x'] = (binx_data['x'] / binx).astype(int) * binx
    binx_data['y'] = (binx_data['y'] / binx).astype(int) * binx
    binx_data = binx_data.groupby(['x', 'y', 'geneID'])['MIDCounts'].sum().to_frame('MIDCounts').reset_index()
    binx_data = binx_data.reindex(columns=['geneID', 'x', 'y', 'MIDCounts'])
    if save is not None:
        if save_format == "gzip":
            binx_data.to_csv(save, index=False, sep='\t', compression="gzip")
        else:
            binx_data.to_csv(save, index=False, sep='\t')
    return binx_data


