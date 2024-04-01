#!/bin/bashsh

sudo apt-get update -y
sudo apt-get install ffmpeg libsm6 libxext6 -y
sudo apt install tmux -y
sudo apt-get install rsync -y
sudo apt-get install libopenexr-dev -y
sudo apt-get install openexr -y

conda install scikit-learn quaternion -y
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y
conda install -c fastai fastprogress -y
conda install -c conda-forge pyyaml -y

yes | pip install --upgrade pip
yes | pip install path
yes | pip install opencv-python
yes | pip install scipy
yes | pip install open3d
yes | pip install imutils
yes | pip install lightning
yes | pip install --upgrade --quiet objaverse

cd "./croco/models/curope/"
python setup.py install
cd "../../../"
