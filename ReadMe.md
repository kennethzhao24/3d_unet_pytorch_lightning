# 3D U-Net with PyTorch-Lightning

This repository includes Pytorch-Lightning implementation of 3D U-Net. Current repo supports both 10045 and shrec2020 dataset.
### Requirements
Pytorch, Pytorch-lightning, visdom, skimage, pycm, prettytable

### Installation
```
git clone http://10.0.5.61/cbmi/3dpp/3dunet-pytorch-lightning.git
```

### Usage
```
cd 3dunet-pytorch-lightning
```

To train your model, run
```
python train.py 
```
To test your trained model, run
```
python test.py
```
 - For oneclass model, it will generate mrcfile and output mean segmentation metrics
 - For multiclass model, it will generate mrcfile mask for each class, output segmentation metrics for each class and mean values
 
For oneclass detection evaluation, run
```
python eval_oneclass.py 
```

For multiclass detection evaluation, run
```
python detection.py
python eval_multiclass.py
```
Detailed results for detection will be generated

For transfer learning, run
```
python -m visdom.server
python re_training.py
```
 - You can load your own checkpoints and model to further modify your re-training schedule


### Shrec20 Results

Model | Precision | Recall | Miss Rate | Avg Distance | F1 Score| 
--- | --- | --- | --- |--- |--- |
Oneclass | 0.8884 | 0.9159 | 0.1116 | 2.2168 | 0.9019 |
Multiclass | 0.8127 | 0.9371 | 0.0629 | 2.3220 | 0.8705 |


### Reference
- [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet)
- [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)
- [SHREC 2020: Classification in cryo-electron tomograms](https://www.sciencedirect.com/science/article/pii/S0097849320301126)