# EfficientSCI
This repo is the implementation of [EfficientSCI: Densely Connected Network with Space-time Factorization for
Large-scale Video Snapshot Compressive Imaging](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_EfficientSCI_Densely_Connected_Network_With_Space-Time_Factorization_for_Large-Scale_Video_CVPR_2023_paper.html).

## Testing Result on Simulation Dataset
<div align="center">
  <img src="docs/psnr_time.png" width=60% />  
  
  Fig1. Comparison of reconstruction quality and testing time of several SOTA deep learning based algorithms.
</div>

## Installation
Please see the [Installation Manual](docs/install.md) for EfficientSCI Installation. 

## Training 
Support multi GPUs and single GPU training efficiently. First download DAVIS 2017 dataset from [DAVIS website](https://davischallenge.org/), then modify *data_root* value in *configs/\_base_/davis.py* file, make sure *data_root* link to your training dataset path.

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/EfficientSCI/efficientsci_base.py --distributed=True
```

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/EfficientSCI/efficientsci_base.py
```

## Testing STFormer on Grayscale Simulation Dataset 
Specify the path of weight parameters, then launch 6 benchmark test in grayscale simulation dataset by executing the statement below.

```
python tools/test.py configs/EfficientSCI/efficientsci_base.py --weights=checkpoints/efficientsci_base.pth
```


## Citation

```
@inproceedings{wang2023efficientsci,
  title={Efficientsci: Densely connected network with space-time factorization for large-scale video snapshot compressive imaging},
  author={Wang, Lishun and Cao, Miao and Yuan, Xin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18477--18486},
  year={2023}
}
```
