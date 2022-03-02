# stDrosophila-release


## ⬇️ Installation
- Create a python virtual env (strongly recommend):
  ```
  conda create -n new_env
  ```
- Install previous versions of PyTorch (recommended torch==1.10.1+cu113):
  ```
  conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  ```
- Install previous versions of PyG (The version of PyG should match the version of pytorch):
  ```
  pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
  ```
- Download this repo and install it:
  ```
  git clone https://github.com/Yao-14/stDrosophila-release.git
  cd ./stDrosophila-release
  pip install .
  ```