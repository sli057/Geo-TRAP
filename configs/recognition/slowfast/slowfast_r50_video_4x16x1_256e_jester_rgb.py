model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained='checkpoints/slowfast_r50_video_4x16x1_256e_kinetics400_rgb_20200826-f85b90c5.pth',  #  None,
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=27,  # 400,
        spatial_type='avg',
        dropout_ratio=0.5))
train_cfg = None
test_cfg = dict(average_clips='prob')
dataset_type = 'RawframeDataset'
data_root = 'data/jester/rawframes'  # changed
data_root_val = 'data/jester/rawframes'  # changed
# split = 1  # official train/test splits. valid numbers: 1, 2, 3  # changed
ann_file_train = f'data/jester/jester_train_list_rawframes.txt'  # changed
ann_file_val = f'data/jester/jester_val_list_rawframes.txt'  # changed
ann_file_test = f'data/jester_val_list_rawframes.txt'  # changed
img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False) # keep it?
#    dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    # dict(type='DecordInit'), # comment out
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1), #clip_len=32, frame_interval=2,
    dict(type='RawFrameDecode'), # dict(type='DecordDecode'),
    dict(type='Resize', scale=(128, -1)),  # dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=112),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    # dict(type='DecordInit'), comment out
    dict(
        type='SampleFrames',
        clip_len=16,  # 32
        frame_interval=1,  # 2
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'), # dict(type='DecordDecode'),
    dict(type='Resize', scale=(128, -1)),  # dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=112),  # dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,  # 32
        frame_interval=1,  # 2
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode'), # dict(type='DecordDecode'),
    dict(type='Resize', scale=(128, -1)),  # dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=112),  # dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,#8,
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
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',  # added
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.001, momentum=0.9, # for 8GPUs * 8 video/GPU lr=0.1
    weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=34)
total_epochs = 15
checkpoint_config = dict(interval=1) # interval=4

evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        #    dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/slowfast_r50_video_3d_4x16x1_256e_jester_no_flip_rgb'
load_from = None
resume_from = None
find_unused_parameters = False
workflow = [('train', 1), ('val', 1)] # added
