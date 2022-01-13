"""
基于拍照图像进行精细抠图：
1. 通过 pre_photo 和 pre_lasso() 分别预处理拍照图像和 lasso 图像
2. 通过 photoshop 配准图像并剪切出需要的图像位置
3. 通过 cropbyphoto 用photoshop后的图像处理 lasso 矩阵
"""

import numpy as np
import pandas as pd
import cv2


def rectangle_crop_img(src, x_start, x_end, y_start, y_end):
    """
    Getting the parts needed in an image by clipping an image with a rectangle.
    Image ( 0, 0 ) points in the upper left corner of the image,
    x_start and x_end determine the height of the image,
    y_start and y_end determine the width of the image.
    """
    return src[x_start: x_end, y_start:y_end]


def flip_img(src, filp_method=1):
    """
    Methods of flipping images,including:
        flipping along X axis ( filp _ method = 0 ),
        flipping along Y axis ( filp _ method = 1 ),
        flipping along XY diagonal ( filp _ method = -1 ).
    """
    return cv2.flip(src, filp_method)


def equalhist_transfer(src, method="global", cliplimit=20):
    """
    Histogram equalization for image enhancement, including:
        global histogram equalization,
        adaptive local histogram equalization.
    """
    if method == "global":
        return cv2.equalizeHist(src)
    elif method == "local":
        clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(7, 7))
        return clahe.apply(src)
    else:
        avail_methods = {"global", "local"}
        raise ValueError(f"Type of histogram equalization method must be one of {avail_methods}.")


def output_img(img, filename, img_name="Image", window_size=(1000, 1000), show_img=True):
    """Output image file.
    """
    if show_img:
        cv2.namedWindow(img_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(img_name, window_size[0], window_size[1])
        cv2.imshow(img_name, img)
    if filename is not None:
        cv2.imwrite(filename=filename, img=img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pre_photo(img,
              rectangle_crop=None,
              coords_flip=None,
              ehtranfer=None,
              color_flip=False,
              gray_factor=None,
              show=True,
              save=None):
    """
    Preprocessing of original image.

    Parameters
    ----------
    img:
        Original photograph (grayscale).
    rectangle_crop: `list` (default: `None`)
        If rectangle_crop is None, the image is not cropped.
        If rectangle_crop is a list of four elements, the image is cropped. e.g., rectangle_crop=[0,100,100,200].
    coords_flip: `int` (default: `None`)
        If coords_flip is None, the image is not flipped.
        If coords_flip is 0, flipping image along X axis,
        If coords_flip is 1, flipping image along Y axis,
        If coords_flip is -1, flipping image along XY diagonal.
    ehtranfer: `dict` (default: `None`)
        The related parameters of histogram equalization.
        e.g., ehtranfer={"method": "local", "cliplimit": 15}:
                The method includes `global` and `local` ,
                the cliplimit refers to the threshold used to limit contrast when method is `local`.
    color_flip: `bool` (default: `False`)
        If color_flip is True, flip the color of the image.
    gray_factor: `int` (default: `1`)
        Increasing the value in the grayscale image.
    show: `bool` (default: `True`)
        If show is True, generate a visual window to display the image.
    save: `str` (default: `None`)
        If save is not None, save is the path to save the image.

    Examples
    --------
    >> photo_img = cv2.imread("E:\BGI_Paper\Data\lasso_ssDNA_20211230\E5\FP200000489TL_E5.tif", 2)
    >> pre_photo(img=photo_img, rectangle_crop=[18500, 20500, 11500, 13500], coords_flip=0,
                 ehtranfer={"method": "local", "cliplimit": 15}, show=True, save="E5_S3_photo.tif")
    """
    img = img.astype("uint8")

    if gray_factor is not None:
        img = img * gray_factor
        img[img > 255] = 255

    if rectangle_crop is not None:
        img = rectangle_crop_img(src=img, x_start=rectangle_crop[0], x_end=rectangle_crop[1],
                                 y_start=rectangle_crop[2], y_end=rectangle_crop[3])

    if ehtranfer is None:
        ehtranfer = {"method": "local", "cliplimit": 15}
    img = equalhist_transfer(img, method=ehtranfer["method"], cliplimit=ehtranfer["cliplimit"])

    if color_flip is True:
        img = cv2.bitwise_not(src=img)

    if coords_flip is not None:
        img = flip_img(src=img, filp_method=coords_flip)

    output_img(img=img, filename=save, img_name="Image", window_size=(1000, 1000), show_img=show)

    return img


def filter_coords(raw_lasso, filter_mtx):
    filter_mtx['y'] = filter_mtx.index
    lasso_data = pd.melt(filter_mtx, id_vars=['y'], value_name="MIDCounts")
    lasso_data = lasso_data[lasso_data["MIDCounts"] != 0][["x", "y"]]
    new_coords = lasso_data["x"].astype(str) + "_" + lasso_data["y"].astype(str)
    new_coords = new_coords.tolist()
    raw_lasso["coords"] = raw_lasso["x"].astype(str) + "_" + raw_lasso["y"].astype(str)
    raw_lasso = raw_lasso[raw_lasso["coords"].isin(new_coords)]
    new_lasso = raw_lasso.drop(columns=["coords"])
    return new_lasso


def pre_lasso(data,
              rectangle_crop=None,
              ehtranfer=None,
              color_flip=False,
              gray_factor=20,
              show=True,
              save_img=None,
              save_lasso=None):
    """
    Preprocessing of original lasso data.

    Parameters
    ----------
    data:
        Original lasso data.
    rectangle_crop: `list` (default: `None`)
        If rectangle_crop is None, the image is not cropped.
        If rectangle_crop is a list of four elements, the image is cropped. e.g., rectangle_crop=[0,100,100,200].
    ehtranfer: `dict` (default: `None`)
        The related parameters of histogram equalization.
        e.g., ehtranfer={"method": "local", "cliplimit": 15}:
                The method includes `global` and `local` ,
                the cliplimit refers to the threshold used to limit contrast when method is `local`.
    color_flip: `bool` (default: `False`)
        If color_flip is True, flip the color of the image.
    gray_factor: `int` (default: `1`)
        Increasing the value in the grayscale image.
    show: `bool` (default: `True`)
        If show is True, generate a visual window to display the image.
    save_img: `str` (default: `None`)
        If save is not None, save is the path to save the image.
    save_lasso: `str` (default: `None`)
        If save is not None, save is the path to save the lasso data.

    Examples
    --------
    >> raw_lasso = pd.read_csv("E:\BGI_Paper\Data\lasso_ssDNA_20211230\E5\E5_S3\lasso\E5_S3_bin5.txt", sep="\t")
    >> new_lasso, img = pre_lasso(data=raw_lasso, rectangle_crop=[200, 500, 500, 800],
                                  ehtranfer={"method": "local", "cliplimit": 15}, gray_factor=3,
                                  show=True, save_img="E5_S3_lasso.tif", save_lasso="E5_S3_lasso.txt")
    """

    raw_lasso = data.copy()
    data = data[["x", "y", "MIDCounts"]].groupby(["x", "y"])["MIDCounts"].sum().to_frame("MIDCounts").reset_index()
    lasso_mtx = pd.pivot_table(data, index=["y"], columns=["x"], values="MIDCounts", fill_value=0)
    if rectangle_crop is not None:
        lasso_mtx = lasso_mtx.iloc[rectangle_crop[0]:rectangle_crop[1], rectangle_crop[2]:rectangle_crop[3]]
    new_lasso = filter_coords(raw_lasso=raw_lasso, filter_mtx=lasso_mtx)
    if save_lasso is not None:
        new_lasso.to_csv(save_lasso, sep="\t", index=False)

    img = pre_photo(lasso_mtx.values, ehtranfer=ehtranfer, color_flip=color_flip,
                    gray_factor=gray_factor, show=show, save=save_img)
    return new_lasso, img


def cropbyphoto(data, img, background=0, save=None):
    """
    Cropping based on photo.

    Parameters
    ----------
    data:
        lasso data generated by pre_lasso().
    img:
        Image after photoshop processing (grayscale).
    background:  `int` (default: `0`)
        Image background color, black is 0, white is 255.
    save: `str` (default: `None`)
        If save is not None, save is the path to save the lasso data.

    Examples
    --------
    >> lasso = pd.read_csv(r"D:\BGIpy37_pytorch113\ST_Drosophila\1_Preprocessing\LassoByImage\E5_S3_lasso.txt", sep="\t")
    >> img = cv2.imread("E5_S3_lasso_ps.tif", 2)
    >> new_lasso = cropbyphoto(data=lasso, img=img, background=255, save="E5_S3_crop.txt")
    """
    raw_lasso = data.copy()
    data = data[["x", "y", "MIDCounts"]].groupby(["x", "y"])["MIDCounts"].sum().to_frame("MIDCounts").reset_index()
    raw_mtx = pd.pivot_table(data, index=["y"], columns=["x"], values="MIDCounts", fill_value=0)
    img[img == background] = 0
    new_mtx = pd.DataFrame(img, index=raw_mtx.index, columns=raw_mtx.columns)
    new_lasso = filter_coords(raw_lasso=raw_lasso, filter_mtx=new_mtx)
    if save is not None:
        new_lasso.to_csv(save, sep="\t", index=False)
    return new_lasso
