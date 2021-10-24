# Geo-TRAP

## Installation
Official pytorch implementation of [NeurIPS 2021 paper Geo-TRAP](https://arxiv.org/abs/2110.01823).

## How to run attack


```shell
python query_attack/decompose_query.py 
    --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py                          
    --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py                          
    --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth                                                  
    --transform_type_query  translation_dilation
```
See installation instruction [here](https://github.com/sli057/Geo-TRAP/blob/main/install.md).


See example adversarial attack commands [here](https://github.com/sli057/Geo-TRAP/blob/main/tools/query_commands.bash)



