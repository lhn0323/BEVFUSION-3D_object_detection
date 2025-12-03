_base_ = [
    './lidar_custom.py'
]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None
image_size = [384, 704]

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  "/home/liujie/code/ADML3D/data/pre-model/swint-nuimages-pretrained.pth" # noqa: E501
        )
        #init_cfg= None
        ),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=image_size,
        feature_size=[48, 88],       # NOTE(Itachi): If you change the image size, you also need to change this. feature_size=image_size/8
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],
        downsample=2),
    fusion_layer=dict(
        type='ConvFuser', in_channels=[80, 256], out_channels=256))

#里面的每一项都会按顺序执行（除了被概率参数控制而不生效的操作），且顺序非常关键
train_pipeline = [
    #第一阶段：数据加载 (Loading)
    #根据 img_path 读取多视角相机图片。新增：字典里多出了 img (图片像素数据) 及其形状信息
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    #根据 lidar_path 读取点云文件。新增：字典里多出了 points (点云坐标/强度数据)
    dict(
        type='CustomLoadPointsFromPCDFile',
        coord_type='LIDAR',
        load_dim=_base_.point_load_dim,
        use_dim=_base_.point_use_dim,
        backend_args=backend_args),
    #作用是将当前关键帧（Key Frame）之前的一系列历史帧（Sweeps）的点云数据加载进来，并将其转换到当前帧的坐标系中，与当前帧的点云合并
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     load_dim=5,
    #     use_dim=5,
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     backend_args=backend_args),
    #读取真值标签（3D 框和类别）。新增：字典里多出了 gt_bboxes_3d 和 gt_labels_3d
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    #第二阶段：数据增强 (Augmentation)
    #ImageAug3D 对图像进行 Resize、裁剪、旋转,它不仅改变图片像素，还会修改相机内参矩阵。如果这里改了顺序，会导致点云投影到图像时对不齐。
    dict(
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    #同时旋转/缩放/平移 点云(points) 和 真值框(gt_bboxes_3d),增加场景的多样性，模拟车辆在不同角度、位置的情况
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='BEVFusionRandomFlip3D'),   #随机翻转点云和框（通常是沿着 X 或 Y 轴翻转）
    #在经过旋转和翻转后，把跑出设定范围（point_cloud_range）的点和框删掉。必须放在几何增强（旋转/平移）之后
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    #把不属于感兴趣类别（如 classes 列表之外）的物体过滤掉
    dict(
        type='ObjectNameFilter',
        classes=_base_.class_names),
    # Actually, 'GridMask' is not used here
    #这意味着它被执行了，但没有生效。程序会判断概率为 0，直接跳过掩码操作。如果你想启用，需要把 prob 改大（例如 0.7）
    dict(
        type='GridMask',
        use_h=True,
        use_w=True,
        max_epoch=6,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.0,
        fixed_prob=True),
    dict(type='PointShuffle'), #打乱点云中点的顺序。因为 PointNet/VoxelNet 类网络对点的输入顺序不敏感，但打乱有助于训练稳定性
    #第三阶段：数据打包 (Formatting)这是最后一步，为了适应 PyTorch 模型输入
    #筛选：根据 keys 列表，把需要送入模型的数据（points, img, gt_bboxes_3d 等）挑出来。
    #转化：把 numpy 数组转为 torch.Tensor。
    #元数据：把 meta_keys 里的信息（如变换矩阵 lidar2cam，增强记录 img_aug_matrix 等）打包进 data_samples.metainfo。BEVFusion 极度依赖这些矩阵来进行特征融合
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='CustomLoadPointsFromPCDFile',
        coord_type='LIDAR',
        load_dim=_base_.point_load_dim,
        use_dim=_base_.point_use_dim,
        backend_args=backend_args),
    #作用是将当前关键帧（Key Frame）之前的一系列历史帧（Sweeps）的点云数据加载进来，并将其转换到当前帧的坐标系中，与当前帧的点云合并
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=9,
    #     load_dim=5,
    #     use_dim=5,
    #     pad_empty_sweeps=True,
    #     remove_close=True,
    #     backend_args=backend_args),
    # ----------------- 新增部分 Start -----------------
    # # 必须加这一段，否则验证集无法计算 Loss 或 mAP
    # dict(
    #     type='LoadAnnotations3D',
    #     with_bbox_3d=True,
    #     with_label_3d=True,
    #     with_attr_label=False),
    # # ----------------- 新增部分 End ------------------
    dict(
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    batch_size=1,
    num_workers = 2,
    dataset=dict(
        dataset=dict(pipeline=train_pipeline, modality=input_modality)))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, modality=input_modality))
test_dataloader = val_dataloader

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        # T_max=6,
        # end=6,
        T_max=20,
        end=20,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 1 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        eta_min=0.85 / 0.95,
        begin=0,
        # end=2.4,
        end=8,         # 按照注释逻辑，前 40% 时间 (20 * 0.4 = 8)
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,
        # begin=2.4,
        # end=6,
        begin=8,       # 接上面
        end=20,        # 结束时间
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    # 原来0.0002
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=1))
del _base_.custom_hooks
work_dir = 'work_dirs/bevfusion_lidar_camera'
#load_from = 'work_dirs/bevfusion_lidar/20251120_093952/epoch_10.pth'
# load_from = '/home/zqh/project/autoware-ml/work_dirs/nus_lidar_cam_4dim_load_from_lyft_epoch_2_7class_nodepth_loss/epoch_3.pth'