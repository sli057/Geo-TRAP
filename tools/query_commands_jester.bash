###############################################################
## ================== Decompose Attack ===================== ##
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
                         --transform_type_query  affine



