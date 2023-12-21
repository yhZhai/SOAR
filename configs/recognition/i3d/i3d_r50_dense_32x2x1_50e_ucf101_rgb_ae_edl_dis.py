_base_ = ['./i3d_r50_dense_32x2x1_50e_ucf101_rgb_ae_edl.py']

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
        zero_init_residual=False),
    cls_head=dict(
        type='AEDebiasHead',
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
            total_epochs=50,
        ),
        loss_recon=dict(type='MSELoss', loss_weight=1.0),
        loss_uncnorm=dict(
            type='UncNormLoss',
            loss_weight=1.0,
            evidence_type='exp',
            num_class=101,
            k=1/8,
            sign=1,
        ),
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        freeze_cls=False,
        freeze_decoder=False,
        with_bn=True,
        recon_grad_rev=False,
        recon_grad_rev_alpha=1.0,
        heavy_cls_head=False,
        do_uncnorm=False,
        loss_debias=dict(type="CrossEntropyLoss", loss_weight=1.0),
        num_scene_classes=365,
        scene_grad_rev_alpha=1.0,
        _delete_=True
    ),
    recon_tgt='frame_raw',
    do_median_filter=False,
    median_win_size=15,
    # model training and testing settings
    train_cfg=dict(aux_info=['scene_feature', 'scene_pred'], _delete_=True),
    test_cfg=dict(average_clips='evidence', evidence_type='exp'))

# dataset settings
dataset_type = 'RawframeSceneFeatureDataset'
scene_feature_root = 'data/ucf101_scene_feature'
scene_pred_root = 'data/ucf101_scene_pred'
data = dict(
    train=dict(
        type=dataset_type,
        scene_feature_root=scene_feature_root,
        scene_pred_root=scene_pred_root),
    val=dict(
        type=dataset_type,
        scene_feature_root=scene_feature_root,
        scene_pred_root=scene_pred_root),
    test=dict(
        type=dataset_type,
        scene_feature_root=scene_feature_root,
        scene_pred_root=scene_pred_root))

# runtime settings
work_dir = './work_dirs/i3d_ae_edl_dis/'
