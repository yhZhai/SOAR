_base_ = ['./i3d_r50_32x2x1_100e_kinetics400_rgb.py']

# model
model = dict(
    type='YZRecognizer3D',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=True,
        pretrained='torchvision://resnet50',
        depth=50,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False
    ),
    cls_head=dict(
        type='AEHead',
        num_classes=101,
        in_channels=2048,
        loss_cls=dict(
            type='EvidenceLoss',
            num_classes=101,
            evidence='exp',
            loss_type='log',
            with_kldiv=False,
            with_avuloss=False,
            annealing_method='exp',
            total_epochs=50
        ),
        loss_recon=dict(type='MSELoss', loss_weight=1.0),
        loss_uncnorm=dict(
            type='UncNormLoss',
            loss_weight=1.0,
            evidence_type='exp',
            num_class=101,
            k=1/8,
            sign=1
        ),
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        freeze_cls=False,
        freeze_decoder=False,
        with_bn=True,
        recon_grad_rev=False,
        grad_rev_alpha=1.,
        heavy_cls_head=False,
        do_uncnorm=False,
    ),
    recon_tgt='frame_raw',
    do_median_filter=False,
    median_win_size=15,
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='evidence', evidence_type='exp')
)

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/ucf101/rawframes'
data_root_val = 'data/ucf101/rawframes'
ann_file_train = 'data/ucf101/ucf101_train_split_1_rawframes.txt'
ann_file_val = 'data/ucf101/ucf101_val_split_1_rawframes.txt'
ann_file_test = 'data/ucf101/ucf101_val_list_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DenseSampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=["frame_dir"]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=8,
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
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=10, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/i3d_ae_edl/'
load_from = 'https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb_20200725-24eb54cc.pth'
custom_hooks = [
    dict(type='AnnealEDLWeightHook', total_epochs=50)
]

# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 50

# optimizer
optimizer = dict(lr=0.001, nesterov=True)
