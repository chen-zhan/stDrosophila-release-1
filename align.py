
# multi-slices Example (slice_alignment_bigBin):
import os
import anndata as ad
import stDrosophila as sd
import torch
## Enter raw anndata data(slices)
folder = "/media/yao/Elements SE/BGI_Paper/L3_new/L3_b/raw_adata/L3_b_bin20"
files = [os.path.join(root, filename) for root, dirs1, files in os.walk(folder) for filename in files]
files.sort()
slices = [ad.read(file) for file in files]

## Enter raw anndata data(slices_big)
folder_big = "/media/yao/Elements SE/BGI_Paper/L3_new/L3_b/raw_adata/L3_b_bin100"
files_big = [os.path.join(root, filename) for root, dirs2, files in os.walk(folder_big) for filename in files]
files_big.sort()
slices_big = [ad.read(file) for file in files_big]

## Alignment
align_slices, align_slices_big = sd.tl.slice_alignment_bigBin(slices=slices, slices_big=slices_big, alpha=0.1, numItermax=200,
                                                              numItermaxEmd=1000000, device=torch.device("cuda:0"), verbose=True)

## Save the slices after alignment
opath = "/media/yao/Elements SE/BGI_Paper/L3_new/L3_b/align_adata/L3_b_bin20"
if not os.path.exists(opath):
    os.mkdir(opath)
for slice in align_slices:
    subSave = os.path.join(opath, f"{slice.obs['slice'][0]}.h5ad")
    slice.write_h5ad(subSave)

## Save the slices_big after alignment
opath_big = "/media/yao/Elements SE/BGI_Paper/L3_new/L3_b/align_adata/L3_b_bin100"
if not os.path.exists(opath_big):
    os.mkdir(opath_big)
for slice in align_slices_big:
    subSave_big = os.path.join(opath_big, f"{slice.obs['slice'][0]}.h5ad")
    slice.write_h5ad(subSave_big)