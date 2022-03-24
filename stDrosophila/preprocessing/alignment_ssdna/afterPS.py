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

from .utils import read_bgi_as_dataframe, output_img


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


def crop_by_ssdna(
    path, path_ssdna, save_lasso=None, save_img=None, show=True
) -> Tuple[pd.DataFrame, np.ndarray]:
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

    img = cv2.imread(path_ssdna, 0)
    background = img[0, 0]

    x_list = np.sort(data["x"].unique())
    y_list = np.sort(data["y"].unique())
    img_table = pd.DataFrame(img, index=x_list, columns=y_list)
    img_table["x"] = img_table.index
    img_data = pd.melt(img_table, id_vars=["x"])
    img_data.columns = ["x", "y", "value"]
    img_data = img_data[img_data["value"] != background]

    cropped_data = pd.merge(data, img_data[["x", "y"]], on=["x", "y"], how="inner")
    cropped_img = pd.pivot_table(
        img_data, index=["x"], columns=["y"], values="value", fill_value=0
    ).values.astype(np.uint8)

    if save_lasso is not None:
        cropped_data.to_csv(save_lasso, sep="\t", index=False)

    if save_img is not None:
        output_img(img=cropped_img, filename=save_img, show_img=show)

    return cropped_data, cropped_img
