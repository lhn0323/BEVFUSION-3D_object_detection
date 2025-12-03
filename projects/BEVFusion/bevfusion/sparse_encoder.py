# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.models.layers import make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.middle_encoders import SparseEncoder
from mmdet3d.registry import MODELS

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor


@MODELS.register_module()
class BEVFusionSparseEncoder(SparseEncoder):
    r"""Sparse encoder for BEVFusion. The difference between this
    implementation and that of ``SparseEncoder`` is that the shape order of 3D
    conv is (H, W, D) in ``BEVFusionSparseEncoder`` rather than (D, H, W) in
    ``SparseEncoder``. This difference comes from the implementation of
    ``voxelization``.
    这意味着 BEVFusionSparseEncoder 内部的 3D 稀疏卷积操作的坐标顺序是 (H, W, D)（即 Y, X, Z),这与它上游的 体素化 实现保持一致，确保了坐标系统的正确匹配

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
        return_middle_feats (bool): Whether output middle features.
            Default to False.
    """

    def __init__(
        self,
        in_channels,
        sparse_shape,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
        return_middle_feats=False,
    ):
        super(SparseEncoder, self).__init__()
        assert block_type in ["conv_module", "basicblock"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        self.return_middle_feats = return_middle_feats
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        if self.order[0] != "conv":  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
                order=("conv",),
            )
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
            )

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule, norm_cfg, self.base_channels, block_type=block_type
        )

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="spconv_down2",
            conv_type="SparseConv3d",
        )

    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Return spatial features
                include:

            - spatial_features (torch.Tensor): Spatial features are out from
                the last layer.
            - encode_features (List[SparseConvTensor], optional): Middle layer
                output features. When self.return_middle_feats is True, the
                module returns middle features.
        """
        coors = coors.int() 
        #voxel_features: 非空体素的特征向量 ([54778, 3])。coors: 非空体素的(B, ZX, Y, Z)坐标。self.sparse_shape: 整个体素网格的逻辑尺寸 [1440, 1440, 41]。batch_size: 批次大小
        input_sp_tensor = SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size) 
        #创建稀疏卷积张量：将输入的体素特征和坐标封装成一个 SparseConvTensor 对象，以便后续的稀疏卷积操作使用。SparseConvTensor[shape=torch.Size([54778, 3])]包含input_sp_tensor.indices[54778, 4]) spatial_shape等
        x = self.conv_input(input_sp_tensor) #初始的稀疏 3D 卷积层，将输入 voxel 特征维度从 3 → 16 x.features.shape:([54778, 16])

        #多层稀疏卷积编码器（配置在 encoder_channels 和 encoder_paddings 中）。
        #功能: 这些层执行特征提取和下采样。在3D空间中进行卷积，使得特征能够捕捉到点云的邻域信息和层次结构。稀疏卷积的优势在于，它只在包含特征数据的非空体素上进行计算，从而大大减少了计算量。
        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x) # 4层 [69635,32] [39161,64] [16677,128] [16677,128]
    
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1]) #稀疏 3D 卷积层，用于将 3D 特征体在 Z 轴上进行聚合或下采样。例如，如果 conv_out 的 kernel_size 是 (1, 1, 3) 且 stride 是 (1, 1, 2)，它只在 XY 平面（BEV 平面）上进行卷积，但在 Z 轴上进行卷积并进行 2 倍下采样
        spatial_features = out.dense() #将稀疏张量转换回密集张量。此时 spatial_features 的维度是 (N, C, H, W, D)[1, 128, 180, 180, 2])（这里N是 Batch Size，D, H, W是稀疏形状中的深度、高度和宽度）
        # out.indices([10601, 4]) out.features([10601, 128]) spatial_shape =[180, 180, 2]
        N, C, H, W, D = spatial_features.shape # [1, 128, 180, 180, 2])
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous() #将维度从 {(N, C, H, W, D)}--> (N, C, D, H, W)([1, 128, 2, 180, 180])
        spatial_features = spatial_features.view(N, C * D, H, W) #将 C 和 D 维度合并，生成最终的 2D BEV 特征图。高度 D上的所有信息都被压缩到通道维度，使得网络可以像处理图像一样处理 BEV 特征
        # spatial_features.shape:([1, 256, 180, 180])
        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
            return spatial_features
