


import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import sklearn
import spateo as st
from scipy import ndimage

plt.rcParams['image.interpolation'] = 'none'


total = st.io.read_bgi_agg("/media/yao/Elements SE/BGI_Paper/E16_18_d/E16_18_d_bin1/raw_lasso/E16-18_d_S10.gem", binsize=1)

for i in total:
    print(i)
scores = st.pp.segmentation.icell.score_pixels(
    total[0],
    k=5,
    method='EM+BP',
    em_kwargs=dict(downsample=100000, seed=2022),
    bp_kwargs=dict(n_threads=8, k=3, square=False, p=2 / 3, q=1 / 3),
)

thresholds = skimage.filters.threshold_multiotsu(scores, classes=3)
est_nuclei_mask = st.pp.segmentation.utils.apply_threshold(scores, 7, thresholds[0])

fig1, axes = plt.subplots(ncols=3, figsize=(9, 3), tight_layout=True)
axes[1].imshow(scores)
axes[1].set_title('scores')

axes[2].imshow(est_nuclei_mask)
axes[2].set_title('nuclei segmentation')
plt.show()
est_marker_mask = st.pp.segmentation.utils.safe_erode(
    scores, 3, square=False, min_area=100, n_iter=10, float_k=5, float_threshold=thresholds[1]
)
est_watershed = st.pp.segmentation.label.watershed(
    total[0], est_nuclei_mask, est_marker_mask, 9
)

fig2, axes = plt.subplots(ncols=2, figsize=(6, 3), tight_layout=True)
axes[0].imshow(est_nuclei_mask)
axes[0].imshow(est_marker_mask, alpha=0.5)
axes[0].set_title('markers')

axes[1].imshow(skimage.color.label2rgb(est_watershed, bg_label=0))
axes[1].set_title('final segmentation')
plt.show()