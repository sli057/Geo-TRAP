###############################################################
## ============ ME Sampler Attack, Baselines =============== ##
###############################################################
# c3d
python query_attack/ME_Sampler.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py \
                         --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth \
                         --work-dir ME_Sampler_untargeted_version1
# slowfast                         
python query_attack/ME_Sampler.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_jester_rgb.py \
                         --checkpoint_rec work_dirs/slowfast_r50_video_3d_4x16x1_256e_jester_no_flip_rgb/epoch_15.pth \
                         --work-dir ME_Sampler_untargeted_version1
# tpn                         
python query_attack/ME_Sampler.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_jester_rgb.py \
                         --checkpoint_rec work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_jester_rgb/epoch_14.pth \
                         --work-dir ME_Sampler_untargeted_version1
# i3d
python query_attack/ME_Sampler.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/i3d/i3d_r50_32x2x1_100e_jester_rgb.py \
                         --checkpoint_rec work_dirs/i3d_r50_32x2x1_100e_jester_rgb/epoch_75.pth \
                         --work-dir ME_Sampler_untargeted_version1
# transfer rate, transfer to slowfast
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
## ========== Heuristic Query Attack, Baselines ============ ##
###############################################################
python query_attack/heuristic_query.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py \
                         --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth \
                         --work-dir Heuristic_untargeted_version1

python query_attack/heuristic_query.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_jester_rgb.py \
                         --checkpoint_rec work_dirs/slowfast_r50_video_3d_4x16x1_256e_jester_no_flip_rgb/epoch_15.pth \
                         --work-dir Heuristic_targeted_version1 --targeted


###############################################################
## ================== Decompose Attack ===================== ##
###############################################################
python query_attack/decompose_query.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py \
                         --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth \
                         --work-dir decompose_query_version1 \
                         --transform_type_query  affine

python query_attack/decompose_query.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py \
                         --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth \
                         --work-dir decompose_query_version1 \
                         --transform_type_query  similarity

python query_attack/decompose_query.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py \
                         --checkpoint_rec work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth \
                         --work-dir decompose_query_version1 \
                         --transform_type_query  projective

python query_attack/plot_loss_curves.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_jester_rgb.py  \
                         --checkpoint_rec work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_jester_rgb/epoch_14.pth \
                         --work-dir decompose_query_version1 \
                         --transform_type_query  translation_dilation \
                         --loss_name flicker_loss \
                         --loss_name cw2_loss \
                         --loss_name cross_entropy_loss \
                         --targeted \

python query_attack/visualization.py --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                         --config_rec configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_jester_rgb.py \
                         --checkpoint_rec work_dirs/slowfast_r50_video_3d_4x16x1_256e_jester_no_flip_rgb/epoch_15.pth \
                         --work-dir decompose_query_version1 \
                         --transform_type_query  translation_dilation

###############################################################
## ============ Hybrid Attack =============== ##
###############################################################
# transfer rate, transfer to slowfast
                                --config_rec_test configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_jester_rgb.py \
                                --checkpoint_rec_test work_dirs/slowfast_r50_video_3d_4x16x1_256e_jester_no_flip_rgb/epoch_15.pth
#                              c3d
                                --config_rec_test configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py \
                                --checkpoint_rec_test work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth
#                              tsm
                                --config_rec_test configs/recognition/tsm/tsm_r50_1x1x16_50e_jester_rgb.py \
                                --checkpoint_rec_test work_dirs/tsm_r50_1x1x16_50e_jester_rgb/epoch_2.pth
#                              tpn
                                --config_rec_test configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_jester_rgb.py \
                                --checkpoint_rec_test work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_jester_rgb/epoch_14.pth
#                              csn
                                --config_rec_test configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_jester_rgb.py \
                                --checkpoint_rec_test work_dirs/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_jester_rgb/epoch_8.pth

#                              i3d
                                --config_rec_test configs/recognition/i3d/i3d_r50_32x2x1_100e_jester_rgb.py \
                                --checkpoint_rec_test work_dirs/i3d_r50_32x2x1_100e_jester_rgb/epoch_75.pth

# from c3d
python query_attack/hybrid_query.py  --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                                --config_rec_train configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py \
                                --work_dir Affine_similarity_version4 \
                                --transform_type similarity \
                                --resume_epoch 10 \
                                --config_rec_test configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_jester_rgb.py \
                                --checkpoint_rec_test work_dirs/slowfast_r50_video_3d_4x16x1_256e_jester_no_flip_rgb/epoch_15.pth

# from slowfast
python query_attack/hybrid_query.py  --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                                --config_rec_train configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_jester_rgb.py \
                                --work_dir Affine_similarity_version4 \
                                --transform_type  similarity \
                                --resume_epoch 10 \
                                --config_rec_test configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_jester_rgb.py \
                                --checkpoint_rec_test work_dirs/slowfast_r50_video_3d_4x16x1_256e_jester_no_flip_rgb/epoch_15.pth

python query_attack/hybrid_query.py  --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                                --config_rec_train configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_jester_rgb.py \
                                --work_dir Affine_similarity_version4 \
                                --transform_type  similarity \
                                --resume_epoch 10 \
                                --config_rec_test configs/recognition/c3d/c3d_sports1m_16x1x1_45e_jester_rgb.py \
                                --checkpoint_rec_test work_dirs/c3d_sports1m_16x1x1_45e_jester_rgb/epoch_30.pth \
                                --update_static 0 \
                                --transform_type_query  affine

# from tpn
python query_attack/hybrid_query.py  --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                                --config_rec_train configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_jester_rgb.py \
                                --work_dir Affine_similarity_version4 \
                                --transform_type  similarity \
                                --resume_epoch 10 \
                                --config_rec_test configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_jester_rgb.py \
                                --checkpoint_rec_test work_dirs/slowfast_r50_video_3d_4x16x1_256e_jester_no_flip_rgb/epoch_15.pth


# from i3d
python query_attack/hybrid_query.py  --config configs/recognition/c3d/c3d_jester_MotionPerturbation.py \
                                --config_rec_train configs/recognition/i3d/i3d_r50_32x2x1_100e_jester_rgb.py \
                                --work_dir Affine_similarity_version4 \
                                --transform_type  similarity \
                                --resume_epoch 10 \
                                --config_rec_test configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_jester_rgb.py \
                                --checkpoint_rec_test work_dirs/slowfast_r50_video_3d_4x16x1_256e_jester_no_flip_rgb/epoch_15.pth
