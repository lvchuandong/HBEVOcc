_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

# Dataset Config
dataset_name = 'openocc'
eval_metric = 'rayiou'

class_weights = [0.0682, 0.0823, 0.0671, 0.0594, 0.0732, 0.0806, 0.0680, 0.0762, 0.0675, 0.0633, 0.0521, 0.0644, 0.0557,
                 0.0551, 0.0535, 0.0533, 0.0464]  # openocc

occ_class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                   'traffic_cone', 'barrier', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
                   'vegetation', 'free']

foreground_idx = [occ_class_names.index('car'), occ_class_names.index('truck'), occ_class_names.index('trailer'),
                  occ_class_names.index('bus'), occ_class_names.index('construction_vehicle'),
                  occ_class_names.index('bicycle'),
                  occ_class_names.index('motorcycle'), occ_class_names.index('pedestrian')]

# DataLoader Config
data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)
# Each nuScenes sequence is ~40 keyframes long. Our training procedure samples
# sequences first, then loads frames from the sampled sequence in order
# starting from the first frame. This reduces training step-to-step diversity,
# lowering performance. To increase diversity, we split each training sequence
# in half to ~20 keyframes, and sample these shorter sequences during training.
# During testing, we do not do this splitting.
train_sequences_split_num = 2
test_sequences_split_num = 1

# Running Config
num_gpus = 8
samples_per_gpu = 2
workers_per_gpu = 4
total_epoch = 48
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu))  # total samples: 28130

# Model Config

# forward params
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

downsample_rate = 16
multi_adj_frame_id_cfg = (1, 1 + 8, 1)
forward_numC_Trans = 80

# backward params
grid_config_bevformer = {
    'x': [-40, 40, 0.8],
    'y': [-40, 40, 0.8],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
backward_bev_h = 100
backward_bev_w = 100
backward_bev_z = 16
backward_num_layer = 2
backward_num_heads = 4
backward_numC_Trans = 160
backward_num_points = 16
# _dim_ = backward_numC_Trans * 2
_pos_dim_ = backward_numC_Trans // 2
_ffn_dim_ = backward_numC_Trans * 2
dropout = 0.1
_num_levels_ = 1

# others params
num_classes = len(occ_class_names)  # 0-15->objects, 16->free

history_frame_num = 8

model = dict(
    type='HBEVOcc',
    class_weights=class_weights,
    foreground_idx=foreground_idx,
    history_frame_num=history_frame_num,
    backward_num_layer=backward_num_layer,
    empty_idx=occ_class_names.index('free'),
    num_classes=num_classes,
    use_mask=False,
    save_results=True,
    grid_config_bevformer= grid_config_bevformer,
    eval_sequential=False,
    forward_projection=dict(
        type='BEVDetStereoForwardProjection',
        align_after_view_transfromation=False,
        return_intermediate=True,
        num_adj=len(range(*multi_adj_frame_id_cfg)),
        img_backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=False,
            with_cp=True,
            style='pytorch'),
        img_neck=dict(
            type='CustomFPN',
            in_channels=[1024, 2048],
            out_channels=256,
            num_outs=1,
            start_level=0,
            out_ids=[0]),
        img_view_transformer=dict(
            type='LSSVStereoForwardPorjection',
            grid_config=grid_config,
            input_size=data_config['input_size'],
            in_channels=256,
            out_channels=forward_numC_Trans,
            sid=False,
            collapse_z=True,
            loss_depth_weight=0.05,
            depthnet_cfg=dict(use_dcn=False,
                              aspp_mid_channels=96,
                              stereo=True,
                              bias=5.),
            downsample=downsample_rate),
        img_bev_encoder_backbone=dict(
            type='CustomResNet',
            numC_input=forward_numC_Trans * (len(range(*multi_adj_frame_id_cfg)) + 1),
            num_layer=[2, 2, 2],
            with_cp=False,
            num_channels=[forward_numC_Trans * 2, forward_numC_Trans * 4, forward_numC_Trans * 8],
            stride=[2, 2, 2],
        ),
        img_bev_encoder_neck=dict(
            type='FPN_LSS',
            in_channels=forward_numC_Trans * 8 + forward_numC_Trans * 2,
            out_channels=256),
        pre_process=dict(
            type='CustomResNet',
            numC_input=forward_numC_Trans,
            num_layer=[1, ],
            num_channels=[forward_numC_Trans, ],
            stride=[1, ],
            backbone_output_ids=[0, ]),
    ),
    backward_projection=dict(
        type='BEVFormerBackwardProjection',
        bev_h=backward_bev_h,
        bev_w=backward_bev_w,
        in_channels=backward_numC_Trans,
        out_channels=backward_numC_Trans,
        pc_range=point_cloud_range,
        transformer=dict(
            type='BEVFormer',
            use_cams_embeds=False,
            embed_dims=backward_numC_Trans,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=backward_num_layer,
                use_temporal=False,
                pc_range=point_cloud_range,
                grid_config=grid_config_bevformer,
                data_config=data_config,
                return_intermediate=False,
                predictor_in_channels=backward_numC_Trans,
                predictor_out_channels=backward_numC_Trans,
                predictor_num_calsses=num_classes,
                transformerlayers=dict(
                    type='BEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            dbound=grid_config_bevformer['depth'],
                            dropout=dropout,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=backward_numC_Trans,
                                num_heads=backward_num_heads,
                                num_points=backward_num_points,
                                num_levels=_num_levels_),
                            embed_dims=backward_numC_Trans,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=backward_numC_Trans ,
                        feedforward_channels=_ffn_dim_,
                        ffn_drop=dropout,
                        act_cfg=dict(type='ReLU', inplace=True),),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=dropout,
                    conv_cfgs=dict(embed_dims=backward_numC_Trans),
                    # operation_order=('predictor', 'self_attn', 'norm', 'cross_attn', 'norm', 'conv')
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm')
                )
            ),
        ),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=backward_bev_h,
            col_num_embed=backward_bev_w,
        ),
    ),
    occupancy_head=dict(
        type='OccHead',
        in_channels=512,
        out_channels=512,
        num_classes=num_classes,
        foreground_idx=foreground_idx,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
    ),
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='PrepareImageInputs', is_train=True, data_config=data_config, sequential=True),
    dict(type='LoadAnnotations'),
    dict(type='LoadOccGTFromFileOpenOcc'),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=occ_class_names),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=3, file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=occ_class_names),
    dict(type='Collect3D',
         keys=['img_inputs', 'gt_depth', 'voxel_semantics', 'voxel_flows'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=occ_class_names, is_train=False),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=3, file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=occ_class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=occ_class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    use_sequence_group_flag=True,
    # Eval Config
    dataset_name=dataset_name,
    eval_metric=eval_metric,
    eval_show=False,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'hbevocc-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'hbevocc-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=occ_class_names,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR',
        # Video Sequence
        sequences_split_num=train_sequences_split_num,
        use_sequence_group_flag=True,
        # Set BEV Augmentation for the same sequence
        # bda_aug_conf=bda_aug_conf,
    ),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
lr = 2e-4
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[num_iters_per_epoch * total_epoch, ])

checkpoint_epoch_interval = 1
runner = dict(type='IterBasedRunner', max_iters=total_epoch * num_iters_per_epoch)
checkpoint_config = dict(interval=checkpoint_epoch_interval * num_iters_per_epoch)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=2 * num_iters_per_epoch,
    )
]

load_from = "ckpts/bevdet-r50-4dlongterm-stereo-cbgs.pth"
revise_keys = [(r'^', 'forward_projection.')]
