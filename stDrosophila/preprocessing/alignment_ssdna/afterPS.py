"""
基于ssDNA进行精细抠图：
1. 通过 lasso_pre() 预处理 lasso 图像
2. 通过 photoshop 配准图像并剪切出需要的图像位置
3. 通过 crop_by_ssdna 用photoshop后的图像处理 lasso 矩阵
"""

import cv2
import pandas as pd
import numpy as np

from typing import Tuple


def read_bgi_as_dataframe(path: str) -> pd.DataFrame:
    """Read a BGI read file as a pandas DataFrame.

    Args:
        path: Path to read file.

    Returns:
        Pandas Dataframe with column names `gene`, `x`, `y`, `total` and
        additionally `spliced` and `unspliced` if splicing counts are present.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype={
            "geneID": "category",  # geneID
            "x": np.uint32,  # x
            "y": np.uint32,  # y
            "MIDCounts": np.uint16,  # total
        },
        comment="#",
    )


def ssdna_resize(img, new_size=None):
    """
    Scale the image, change the size of the image.

    Args:
        img: Image matrix.
        new_size: The size of the scaled image.
    Returns:
        A scaled image matrix.
    """

    return cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)


def output_img(
    img: np.ndarray,
    filename: str = None,
    window_size: tuple = (1024, 1024),
    show_img: bool = True
):
    """
    Output the image matrix as a 2D image file.

    Args:
        img：Image matrix.
        filename: Output image filename, the end of which can be .bmp, .dib, .jpeg, .jpg, .jpe, .png, .webp, .pbm,
                  .pgm, .ppm, .pxm, .pnm, .sr, .ras, .tiff, .tif, .exr, .hdr, .pic, etc.
        window_size: The size of the image visualization window.
        show_img: Whether to create a window to display the image.
    """

    if show_img:
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("Image", window_size[0], window_size[1])
        cv2.imshow("Image", img)
    if filename is not None:
        cv2.imwrite(filename=filename, img=img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop_by_ssdna(path, path_ssdna, save_lasso=None, save_img=None, show=True) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Cropping based on ssDNA.

    Args:
        path: Path to read lasso file.
        path_ssdna: Path to read ssDNA image file (grayscale).
        save_lasso: If save is not None, save is the path to save the lasso data.
        save_img: If save is not None, save is the path to save the image.
        show: If show is True, generate a visual window to display the image.

    """
    data = read_bgi_as_dataframe(path=path)

    img = cv2.imread(path_ssdna, 2)
    background = img[0, 0]

    x_list = np.sort(data["x"].unique())
    y_list = np.sort(data["y"].unique())
    img_table = pd.DataFrame(img, index=y_list, columns=x_list)
    img_table["y"] = img_table.index
    img_data = pd.melt(img_table, id_vars=["y"])
    img_data.columns = ["y", "x", "value"]
    img_data = img_data[img_data["value"] != background]

    cropped_data = pd.merge(data, img_data[["x", "y"]], on=["x", "y"], how="inner")
    cropped_img = pd.pivot_table(
        img_data, index=["y"], columns=["x"], values="value", fill_value=0
    ).values.astype(np.uint8)

    if save_lasso is not None:
        cropped_data.to_csv(save_lasso, sep="\t", index=False)

    if save_img is not None:
        output_img(img=cropped_img, filename=save_img, show_img=show)

    return cropped_data, cropped_img
