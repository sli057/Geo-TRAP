###############################################################
## ============ ME Sampler Attack, Baselines =============== ##
###############################################################
# c3d
python query_frame/ME_Sampler.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                                 --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
                                 --checkpoint_rec  checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth \
                                 --work-dir ME_Sampler_untargeted_version1
# slowfast
python query_frame/ME_Sampler.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                                 --config_rec configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_ucf_rgb.py \
                                 --checkpoint_rec work_dirs/slowfast_r50_video_3d_4x16x1_256e_ucf101_1_rgb/epoch_30.pth \
                                 --work-dir ME_Sampler_untargeted_version1
# tpn
python query_frame/ME_Sampler.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                                 --config_rec configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf_rgb.py \
                                 --checkpoint_rec work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf101_1_rgb/epoch_145.pth \
                                 --work-dir ME_Sampler_untargeted_version1
# i3d
python query_frame/ME_Sampler.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                                 --config_rec configs/recognition/i3d/i3d_r50_32x2x1_100e_ucf_rgb.py \
                                 --checkpoint_rec work_dirs/i3d_r50_32x2x1_100e_ucf_split_1_rgb/epoch_100.pth \
                                 --work-dir ME_Sampler_untargeted_version1
# transfer rate, transfer to slowfast
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

###############################################################
## ========== Heuristic Query Attack, Baselines ============ ##
###############################################################
python query_attack/heuristic_query.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                                 --config_rec configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_ucf_rgb.py \
                                 --checkpoint_rec work_dirs/slowfast_r50_video_3d_4x16x1_256e_ucf101_1_rgb/epoch_30.pth \
                                 --work-dir Heuristic_untargeted_version1
###############################################################
## ================== Decompose Attack ===================== ##
###############################################################
python query_attack/decompose_query.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                                 --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
                                 --checkpoint_rec  checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth \
                                 --work-dir decompose_query_version1 \
                                 --transform_type_query  affine

python query_attack/plot_loss_curves.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                         --config_rec configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf_rgb.py \
                         --checkpoint_rec  work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf101_1_rgb/epoch_145.pth \
                         --work-dir decompose_query_version1 \
                         --transform_type_query  translation_dilation \
                         --targeted \
                         --loss_name flicker_loss \
                         --loss_name cw2_loss \
                         --loss_name cross_entropy_loss \

python query_attack/visualization.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                         --config_rec configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_ucf_rgb.py \
                         --checkpoint_rec work_dirs/slowfast_r50_video_3d_4x16x1_256e_ucf101_1_rgb/epoch_30.pth \
                         --work-dir decompose_query_version1 \
                         --transform_type_query  translation_dilation

###############################################################
## ============ Hybrid Attack =============== ##
###############################################################
# transfer rate, transfer to slowfast
                                --config_rec_test configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_ucf_rgb.py \
                                --checkpoint_rec_test work_dirs/slowfast_r50_video_3d_4x16x1_256e_ucf101_1_rgb/epoch_30.pth
#                               tsm
                                --config_rec_test configs/recognition/tsm/tsm_r50_1x1x16_50e_ucf_rgb.py \
                                --checkpoint_rec_test work_dirs/tsm_r50_1x1x16_50e_ucf101_1_rgb/epoch_50.pth
#                               tpn
                                --config_rec_test configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf_rgb.py \
                                 --checkpoint_rec_test work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf101_1_rgb/epoch_145.pth

#                               csn
                                --config_rec_test configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_ucf_rgb.py \
                                --checkpoint_rec_test work_dirs/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_ucf101_1_rgb/epoch_50.pth

#                               i3d
                                --config_rec_test configs/recognition/i3d/i3d_r50_32x2x1_100e_ucf_rgb.py \
                                --checkpoint_rec_test work_dirs/i3d_r50_32x2x1_100e_ucf_split_1_rgb/epoch_100.pth
#                               c3d
                                --config_rec_test configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
                                --checkpoint_rec_test checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth

# from c3d
python query_attack/hybrid_query.py  --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                                --config_rec_train configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
                                --work_dir Affine_rigid_version7 \
                                --transform_type rigid \
                                --resume_epoch 250 \
                                --config_rec_test configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
                                --checkpoint_rec_test checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth

# from slowfast
python query_attack/hybrid_query.py  --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                                --config_rec_train configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_ucf_rgb.py \
                                --work_dir Affine_rigid_version7\
                                --transform_type rigid \
                                --resume_epoch 250 \
                                --config_rec_test configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
                                --checkpoint_rec_test checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth


# from tpn
python query_attack/hybrid_query.py  --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                                --config_rec_train configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_ucf_rgb.py  \
                                --work_dir Affine_rigid_version4_small_batch \
                                --transform_type rigid \
                                --resume_epoch 200 \
                                --config_rec_test configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
                                --checkpoint_rec_test checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth


# from i3d
python query_attack/hybrid_query.py --config configs/recognition/c3d/c3d_ucf_MotionPerturbation.py \
                                --config_rec_train configs/recognition/i3d/i3d_r50_32x2x1_100e_ucf_rgb.py  \
                                --work_dir Affine_rigid_version7 \
                                --transform_type rigid \
                                --resume_epoch 250 \
                                --config_rec_test configs/recognition/c3d/c3d_sports1m_16x1x1_45e_ucf101_rgb.py \
                                --checkpoint_rec_test checkpoints/c3d_sports1m_16x1x1_45e_ucf101_rgb_20201021-26655025.pth

