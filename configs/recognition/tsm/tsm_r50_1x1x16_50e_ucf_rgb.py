# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained=  'torchvision://resnet50',  # 'checkpoints/tsm_r50_256p_1x1x16_50e_kinetics400_rgb_20201010-85645c2a.pth',  # 'torchvision://resnet50',
        depth=50,
        num_segments=16,
        norm_eval=False,
        shift_div=8),
    cls_head=dict(
        type='TSMHead',
        num_classes=101,
        num_segments=16,
        in_channels=2048, # ??
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'RawframeDataset'  # 'VideoDataset' changed
data_root = 'data/ucf101/rawframes'  # 'data/kinetics400/videos_train'
data_root_val = 'data/ucf101/rawframes'  # 'data/kinetics400/videos_val'
split = 1  #changed
ann_file_train = f'data/ucf101/ucf101_train_split_{split}_rawframes.txt'  # 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'  # 'data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'  # 'data/kinetics400/kinetics400_val_list_videos.txt'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False) 
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, -1)),  # (-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=112, # 224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),  # we need 112 # (224, 224)
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, -1)),  # (-1, 256)), # we need 128
    dict(type='CenterCrop', crop_size=112),  # 224), # we need 112
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(128, -1)),  # (-1, 256)), # same
    dict(type='CenterCrop', crop_size=112),  # 224), # same
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',  # need exploaration
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.001, # 0.0075,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/tsm_r50_1x1x16_50e_ucf101_{split}_rgb'  # changed
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)] # added
