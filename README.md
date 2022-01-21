# stDrosophila-release



## ⬇️ Installation
- Create a python virtual env (strongly recommend):
  ```
  conda create -n new_env
  ```
- Install previous versions of PyTorch (recommended torch==1.10.1+cu113):
  ```
  pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
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