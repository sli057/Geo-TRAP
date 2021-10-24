###############################################################
## ================== Jester Dataset ===================== ##
###############################################################
python query_attack/decompose_query.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py \
                         --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth \
                         --work-dir decompose_query_version1 \
                         --transform_type_query  translation_dilation

python query_attack/decompose_query.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py \
                         --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth \
                         --work-dir decompose_query_version1 \
                         --transform_type_query  affine --targeted

#                           slowfast
                         --config_rec configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_jester_rgb.py \
                         --checkpoint_rec work_dirs/slowfast_r50_video_3d_4x16x1_256e_jester_no_flip_rgb/epoch_15.pth \
#                              tpn
                         --config_rec configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_jester_rgb.py \
                         --checkpoint_rect work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_jester_rgb/epoch_14.pth \
#                              c3d
                         --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py \
                         --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth \
#                              i3d
                         --config_rec configs/recognition/i3d/i3d_r50_32x2x1_100e_jester_rgb.py \
                         --checkpoint_rec work_dirs/i3d_r50_32x2x1_100e_jester_rgb/epoch_75.pth \


###############################################################
## ================== UCF-101 Dataset ===================== ##
###############################################################

python query_attack/decompose_query.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                                 --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
                                 --checkpoint_rec  checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth \
                                 --work-dir decompose_query_version1 \
                                 --transform_type_query  translation_dilation

#                               slowfast
                                 --config_rec configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_ucf_rgb.py \
                                 --checkpoint_rec work_dirs/slowfast_r50_video_3d_4x16x1_256e_ucf101_1_rgb/epoch_30.pth \
#                                tpn
                                 --config_rec configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf_rgb.py \
                                 --checkpoint_rec work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf101_1_rgb/epoch_145.pth \
#                                i3d
                                 --config_rec configs/recognition/i3d/i3d_r50_32x2x1_100e_ucf_rgb.py \
                                 --checkpoint_rec work_dirs/i3d_r50_32x2x1_100e_ucf_split_1_rgb/epoch_100.pth \
#                                c3d
                                 --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
                                 --checkpoint_rec checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth \


