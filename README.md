# Geo-TRAP
Official pytorch implementation of NeurIPS 2021 paper
[Adversarial Attacks on Black Box Video Classifiers:
Leveraging the Power of Geometric Transformations](https://arxiv.org/abs/2110.01823).


## Installation
See installation instruction [here](install.md).


## Prepare data
For details on data preparation, you can refer to

* [preparing_ucf101](/tools/data/ucf101/README.md)
* [preparing_jester](/tools/data/jester/README.md)

## Pre-trained video models
The pretrained C3D, SlowFast, TPN and I3D model on both UCF-101 and Jester dataset can be found in 
[Dropbox](https://www.dropbox.com/sh/goltkscppr0k53n/AAAkCa34PrRgIk8wYNxx_WJ3a?dl=0).

## Usage

Here, we give an example of how to do targeted attack to C3D model on Jester dataset with affine transformation.
```shell
python query_attack/decompose_query.py 
    --targeted # targeted attack 
    --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py  # contains configuration of Jester dataset                         
    --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py # contains configuration of C3D model                         
    --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth # pretrained C3D model paramters                                            
    --transform_type_query  affine # use affine geometric transformation
```


For untargeted attacks, other video models (SlowFast, TPN and I3D) and UCF101 dataset, please see adversarial attack commands [here](tools/query_commands.bash)
 
 
##  Citation

If you find our work helpful in your research, please cite it as 

```shell
@article{li2021adversarial,
  title={Adversarial Attacks on Black Box Video Classifiers: Leveraging the Power of Geometric Transformations},
  author={Li, Shasha and Aich, Abhishek and Zhu, Shitong and Asif, M Salman and Song, Chengyu and Roy-Chowdhury, Amit K and Krishnamurthy, Srikanth},
  journal={arXiv preprint arXiv:2110.01823},
  year={2021}
}
```

## Acknowledgement

Many thanks to [MMAction2](https://github.com/open-mmlab/mmaction2.git) for the video model implementation.

