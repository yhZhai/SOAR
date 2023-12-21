# model settings
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
        do_uncnorm=False
    ),
    recon_tgt='frame_raw',
    do_median_filter=False,
    median_win_size=15,
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='score')
)

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/ucf101/videos'
data_root_val = 'data/ucf101/videos'
ann_file_train = 'data/ucf101/ucf101_train_split_1_videos.txt'
ann_file_val = 'data/ucf101/ucf101_val_split_1_videos.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
test_pipeline = [
    dict(type="DecordInit"),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type="DecordDecode"),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=["filename"]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=8,
    test=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True))
