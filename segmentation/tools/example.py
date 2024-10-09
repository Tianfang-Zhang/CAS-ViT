norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='../../checkpoint/resnet50_v1c-2cccc1ad.pth',
    backbone=dict(
        type='rcvit_x5_12m',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../../checkpoint/rcvit_x5_12m_79.05.pth')),
    neck=dict(
        type='FPN',
        in_channels=[64, 96, 192, 384],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            backend='petrel',
            path_mapping=dict({
                'data/ade/ADEChallengeData2016/':
                'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                '.data/ade/ADEChallengeData2016/':
                'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
            }))),
    dict(
        type='LoadAnnotations',
        reduce_zero_label=True,
        file_client_args=dict(
            backend='petrel',
            path_mapping=dict({
                'data/ade/ADEChallengeData2016/':
                'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                '.data/ade/ADEChallengeData2016/':
                'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
            }))),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(
            backend='petrel',
            path_mapping=dict({
                'data/ade/ADEChallengeData2016/':
                'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                '.data/ade/ADEChallengeData2016/':
                'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
            }))),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='AlignResize', keep_ratio=True, size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type='ADE20KDataset',
            data_root='data/ade/ADEChallengeData2016/',
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(
                        backend='petrel',
                        path_mapping=dict({
                            'data/ade/ADEChallengeData2016/':
                            'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                            '.data/ade/ADEChallengeData2016/':
                            'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
                        }))),
                dict(
                    type='LoadAnnotations',
                    reduce_zero_label=True,
                    file_client_args=dict(
                        backend='petrel',
                        path_mapping=dict({
                            'data/ade/ADEChallengeData2016/':
                            'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                            '.data/ade/ADEChallengeData2016/':
                            'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
                        }))),
                dict(
                    type='Resize',
                    img_scale=(2048, 512),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(512, 512),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ],
            file_client_args=dict(
                backend='petrel',
                path_mapping=dict({
                    'data/ade/ADEChallengeData2016/':
                    'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                    '.data/ade/ADEChallengeData2016/':
                    'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
                })))),
    val=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016/',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(
                    backend='petrel',
                    path_mapping=dict({
                        'data/ade/ADEChallengeData2016/':
                        'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                        '.data/ade/ADEChallengeData2016/':
                        'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
                    }))),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='AlignResize', keep_ratio=True, size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        gt_seg_map_loader_cfg=dict(
            file_client_args=dict(
                backend='petrel',
                path_mapping=dict({
                    'data/ade/ADEChallengeData2016/':
                    'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                    '.data/ade/ADEChallengeData2016/':
                    'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
                }))),
        file_client_args=dict(
            backend='petrel',
            path_mapping=dict({
                'data/ade/ADEChallengeData2016/':
                'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                '.data/ade/ADEChallengeData2016/':
                'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
            }))),
    test=dict(
        type='ADE20KDataset',
        data_root='data/ade/ADEChallengeData2016/',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(
                    backend='petrel',
                    path_mapping=dict({
                        'data/ade/ADEChallengeData2016/':
                        'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                        '.data/ade/ADEChallengeData2016/':
                        'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
                    }))),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='AlignResize', keep_ratio=True, size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        gt_seg_map_loader_cfg=dict(
            file_client_args=dict(
                backend='petrel',
                path_mapping=dict({
                    'data/ade/ADEChallengeData2016/':
                    'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                    '.data/ade/ADEChallengeData2016/':
                    'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
                }))),
        file_client_args=dict(
            backend='petrel',
            path_mapping=dict({
                'data/ade/ADEChallengeData2016/':
                'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
                '.data/ade/ADEChallengeData2016/':
                'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
            }))))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl', port=29503)
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        'data/ade/ADEChallengeData2016/':
        'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/',
        '.data/ade/ADEChallengeData2016/':
        'openmmlab:s3://openmmlab/datasets/segmentation/ade/ADEChallengeData2016/'
    }))
gpu_multiples = 2
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
