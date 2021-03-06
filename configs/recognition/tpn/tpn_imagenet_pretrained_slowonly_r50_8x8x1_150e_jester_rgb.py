# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained='torchvision://resnet50',
        lateral=False,
        out_indices=(2, 3),
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False),
    neck=dict(
        type='TPN',
        in_channels=(1024, 2048),
        out_channels=1024,
        spatial_modulation_cfg=dict(
            in_channels=(1024, 2048), out_channels=2048),
        temporal_modulation_cfg=dict(downsample_scales=(8, 8)),
        upsample_cfg=dict(scale_factor=(1, 1, 1)),
        downsample_cfg=dict(downsample_scale=(1, 1, 1)),
        level_fusion_cfg=dict(
            in_channels=(1024, 1024),
            mid_channels=(1024, 1024),
            out_channels=2048,
            downsample_scales=((1, 1, 1), (1, 1, 1))),
        aux_head_cfg=dict(out_channels=400, loss_weight=0.5)),
    cls_head=dict(
        type='TPNHead',
        num_classes=27, # changed
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.01))
train_cfg = None
test_cfg = dict(average_clips='prob')
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/jester/rawframes'  # 'data/kinetics400/rawframes_train'
data_root_val = 'data/jester/rawframes'  # 'data/kinetics400/rawframes_val'
ann_file_train = 'data/jester/jester_train_list_rawframes.txt'  # 'data/kinetics400/kinetics400_train_list_rawframes.txt'
ann_file_val = 'data/jester/jester_val_list_rawframes.txt'  # 'data/kinetics400/kinetics400_val_list_rawframes.txt'
ann_file_test = 'data/jester_val_list_rawframes.txt'  # 'data/kinetics400/kinetics400_val_list_rawframes.txt'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False) 
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1), # a quite change clip_len=8, frame_interval=8,
    dict(type='FrameSelector'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(112, 112), keep_ratio=False), # (224, 224) 
    dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,  # 8,
        frame_interval=1,  # 8,
        num_clips=1,
        test_mode=True),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(128, -1)),  # (-1, 256)), # we need 128
    dict(type='CenterCrop', crop_size=112),  # 224), # we need 112
    dict(type='ColorJitter', color_space_aug=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,  # 8,
        frame_interval=1,  # 8,
        num_clips=10,
        test_mode=True),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(128, -1)),  # (-1, 256)), # we need 128
    dict(type='ThreeCrop', crop_size=112),  # type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{:05}.jpg',  # added
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',  # added
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',  # added
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001,
    nesterov=True)  # 0.01 this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[75, 125])
total_epochs = 150
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        #   dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_jester_rgb'  # noqa: E501
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)] # added
