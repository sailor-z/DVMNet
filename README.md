# DVMNet: Computing Relative Pose for Unseen Objects Beyond Hypotheses
PyTorch implementation of "DVMNet: Computing Relative Pose for Unseen Objects Beyond Hypotheses" (CVPR 2024)


[[project page](https://sailor-z.github.io/projects/CVPR2024_DVMNet.html)] &nbsp; &nbsp; &nbsp; &nbsp; [[paper](https://arxiv.org/pdf/2403.13683v1.pdf)]

# Setup Dependencies
```
conda create -n dvmnet python=3.8 cmake=3.14.0
conda activate dvmnet
bash ./install.sh
```
Download the pretrained croco model:
```
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTBase_BaseDecoder.pth -P ./croco/
```
# Data Preparation
Please refer to the instructions provided in [3DAHV](https://github.com/sailor-z/3DAHV) for downloading and preprocessing Co3D, Objaverse, and LINEMOD.

## Test pretrained model
We provide a model pretrained on the training set of CO3D. Please download it [here](https://drive.google.com/file/d/1ENeRWvMNyvYN4QmhtEaAkvLR1FbQstp4/view?usp=sharing). We store this pretrained model at `./models/checkpoint_co3d.ckpt` by default.
Run the following evaluation to get the results:
```
python ./test_co3d_dvmnet.py
```
Notably, the reproduced results might be slightly different from those reported in the paper. This is because the image pairs during testing are randomly sampled in the RelPose++ implementation.

## Trainning
### Co3D
```
python ./train_dvmnet_co3d.py
```
### Objaverse
```
python ./train_dvmnet_objaverse.py
```
### LINEMOD
```
python ./train_dvmnet_linemod.py
```
We also implement a 6D pose estimation model `DVMNet_6D`. The translation estimation module is borrowed from [RelPose++](https://github.com/amyxlase/relpose-plus-plus).

# Citation
If you find the project useful, please consider citing:
```bibtex
@article{zhao2024dvmnet,
  title={DVMNet: Computing Relative Pose for Unseen Objects Beyond Hypotheses},
  author={Zhao, Chen and Zhang, Tong and Dang, Zheng and Salzmann, Mathieu},
  journal={arXiv preprint arXiv:2403.13683},
  year={2024}
}
```
