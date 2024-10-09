_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# data
data = dict(samples_per_gpu=4)

# optimizer
model = dict(
    pretrained=None,
    backbone=dict(
        type='rcvit_s',
        style='pytorch',
        fork_feat=True,
        drop_path_rate=0,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./checkpoint/cas-vit-s.pth',
        )
    ),
    neck=dict(
        type='FPN',
        in_channels=[48, 64, 128, 256],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

find_unused_parameters=True