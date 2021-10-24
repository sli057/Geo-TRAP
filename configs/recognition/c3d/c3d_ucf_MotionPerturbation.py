# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth',  # noqa: E501
        style='pytorch',
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dropout_ratio=0.5,
        init_std=0.005),
    cls_head=dict(
        type='I3DHead',
        num_classes=101,
        in_channels=4096,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01))
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
# dataset settings
dataset_type =  'RawframeRGBFlowDataset'
data_root = 'data/ucf101/rawframes'
data_root_val = 'data/ucf101/rawframes'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'data/ucf101/ucf101_train_split_{split}_rawframes.txt'
ann_file_val = f'data/ucf101/ucf101_val_split_{split}_rawframes.txt'
ann_file_test = f'data/ucf101/ucf101_test_split_{split}_rawframes.txt'
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1),
    dict(type='RawFrameRGBFlowDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112), # changed by Shasha
    dict(type='Flip', flip_ratio=0.5),
    dict(type='NormalizeRGBFlow', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True),#test_mode=True
    # test_model = True:  Calculate the average interval for selected frames, and shift them
    #         fixedly by avg_interval/2.
    # test_model = False: It will calculate the average interval for selected frames,
    #         and randomly shift them within offsets between [0, avg_interval].
    dict(type='RawFrameRGBFlowDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='Flip', flip_ratio=0), # only different from train_pipeline
    dict(type='NormalizeRGBFlow', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=10, # only different from val_pipeline
        test_mode=True),
    dict(type='RawFrameRGBFlowDecode'),
    dict(type='Resize', scale=(128, 171)),
    dict(type='CenterCrop', crop_size=112),
    dict(type='Flip', flip_ratio=0),
    dict(type='NormalizeRGBFlow', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
data = dict(
    videos_per_gpu=15, # with num_clip=1, this means batch_size = 15
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
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
    constructor = 'GeneratorOptimizerConstructor',
    type='SGD', lr=0.001, momentum=0.9, #lr=0.001 
    weight_decay=0.0001)  # this lr is used for 8 gpus weight_decay=0.0005
optimizer_config = dict(
    type='GeneratorOptimizerHook',
    grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[60, 150])
# lr_config = dict(
#     policy='CosineAnnealing',
#     min_lr=0,
#     warmup='linear',
#     warmup_by_epoch=True,
#     warmup_iters=34)
    
total_epochs = 250
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy']) # 5
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/c3d_sports1m_16x1x1_45e_ucf101_split_{split}_rgb/'
load_from = None
resume_from = None
workflow = [('train',5),('val',1)]
