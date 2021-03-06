# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dCSN',
        pretrained2d=False,
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth',  # noqa: E501
        depth=152,
        with_pool2=False,
        bottleneck_mode='ir',
        norm_eval=True,
        bn_frozen=True,
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=27,  # changed 400,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
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
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),  # clip_len=32, frame_interval=2,
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(128, -1)),  # (-1, 256)), # we need 128
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),  # changed scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,  # 32,
        frame_interval=1,  # 2,
        num_clips=1,
        test_mode=True),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(128, -1)),  # (-1, 256)), # we need 128
    dict(type='CenterCrop', crop_size=112),  # c
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16, # 32,
        frame_interval=1, # 2,
        num_clips=10,
        test_mode=True),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(128, -1)),  # (-1, 256)), # we need 128
    dict(type='ThreeCrop', crop_size=112),  # type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=3,
    workers_per_gpu=4,
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
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',  # added
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.000125, momentum=0.9, # not changed
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[32, 48],
    warmup='linear',
    warmup_ratio=0.1,
    warmup_by_epoch=True,
    warmup_iters=16)
total_epochs = 58
checkpoint_config = dict(interval=2)
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_jester_rgb'  # noqa: E501
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)] # added
find_unused_parameters = True
