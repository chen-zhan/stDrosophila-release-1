import cv2
import numpy as np
import pandas as pd


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


def output_img(
    img: np.ndarray,
    filename: str = None,
    window_size: tuple = (1024, 1024),
    show_img: bool = True,
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
    cv2.destroyWindow("Image")
    for i in range(1, 5):
        cv2.waitKey(1)
