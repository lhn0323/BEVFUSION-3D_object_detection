from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from .ops import Voxelization


@MODELS.register_module()
class BEVFusion(Base3DDetector):
    """
     点云 → 体素化 → HardSimpleVFE → SparseEncoder(256C)
     图像 → Swin → FPN → view_transform → BEV(80C)
     融合 → ConvFuser(336→256)
     BEV Backbone SECOND
     BEV Neck SECONDFPN (输出 256C)
     BEV Head BEVFusionHead
    """
    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop("voxelize_cfg")
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop("voxelize_reduce")
        # 把原始点云（N × 4）转成稀疏 voxel 特征 voxelization 层只负责把点云聚簇成 voxel + 记录坐标
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        # 把每个 voxel 内部的点特征编码成一个特征向量
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        # 从多相机 RGB 图像中提取特征backbone输出多层 feature maps swin transformer or resnet
        self.img_backbone = MODELS.build(img_backbone) if img_backbone is not None else None
        # 在 backbone 的多尺度输出上做融合 / 上/下采样（例如 FPN）GeneralizedLSSFPN
        self.img_neck = MODELS.build(img_neck) if img_neck is not None else None
        # 核心模块把图像特征投影到 BEV 基于深度估计 + 相机内参 + 外参将多摄像头 2D 特征 warp 到 BEV 平面上 默认通道 = 80
        self.view_transform = MODELS.build(view_transform) if view_transform is not None else None
        # 点云稀疏体素卷积中间编码器 使用 SparseConvNet 处理稀疏 voxel输出 BEV 特征图 (HxW, C=256)对应 fusion_layer 的输入部分。
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        # 融合模块（图像 BEV + 点云 BEV)
        self.fusion_layer = MODELS.build(fusion_layer) if fusion_layer is not None else None
        # 用 PointPillars/SECOND 的2D CNN进一步提取 BEV 特征
        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
        # 负责检测（分类 + 回归）
        self.bbox_head = MODELS.build(bbox_head)
        # 主要为 img_backbone 加载预训练权重
        self.init_weights()

    def _forward(self, batch_inputs_dict: Tensor, batch_data_samples: OptSampleList = None, **kwargs):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """

        # NOTE(knzo25): this is used during onnx export
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, _ = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head(feats, batch_input_metas)

        return outputs[0][0]

    def parse_losses(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append([loss_name, sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(value for key, value in log_vars if "loss" in key)
        log_vars.insert(0, ["loss", loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    # def init_weights(self) -> None:
    #     if self.img_backbone is not None and self.img_backbone.init_cfg.checkpoint is not None:
    #         self.img_backbone.init_weights()

    def init_weights(self) -> None:
        if self.img_backbone is not None:
        # 安全判断 init_cfg
           init_cfg = getattr(self.img_backbone, 'init_cfg', None)
           if init_cfg is not None and getattr(init_cfg, 'checkpoint', None) is not None:
               self.img_backbone.init_weights()


    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, "bbox_head") and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head."""
        return hasattr(self, "seg_head") and self.seg_head is not None

    def extract_img_feat(
        self,
        x, # 图像 [B, N_cam, 3, H, W]
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        camera_intrinsics_inverse=None,
        img_aug_matrix_inverse=None,
        lidar_aug_matrix_inverse=None,
        geom_feats=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size() # (1, 5, 3, 384, 704)
        x = x.view(B * N, C, H, W).contiguous() # 维度压缩：多相机合并 batch （5, 3, 384, 704)

        x = self.img_backbone(x)# 经 Swin Transformer 提取多尺度特征
        # x[0].shape torch.Size([5, 192, 48, 88]) 1/8
        # x[1].shape torch.Size([5, 384, 24, 44]) 1/16
        # x[2].shape torch.Size([5, 768, 12, 22]) 1/32
        x = self.img_neck(x) # 用 FPN 融合多尺度
        # P3: [5, 256, 48, 88]   (融合 x[0] & x[1])
        # P4: [5, 256, 24, 44]   (融合 x[1] & x[2])


        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size() # [B, 256, 48, 88] 
        assert BN == B * N, (BN, B * N) #
        x = x.view(B, N, C, H, W) # 再拆回多相机维度 torch.Size([1, 5, 256, 48, 88])

        with torch.cuda.amp.autocast(enabled=False):
            # with torch.autocast(device_type='cuda', dtype=torch.float32):
            x, depth_loss = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
                camera_intrinsics_inverse,
                img_aug_matrix_inverse,
                lidar_aug_matrix_inverse,
                geom_feats,
            )
        return x, depth_loss

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:

        if "voxels" not in batch_inputs_dict:
            # NOTE(knzo25): training and normal inference
            points = batch_inputs_dict["points"]
            with torch.cuda.amp.autocast(enabled=False):
                # with torch.autocast('cuda', enabled=False):
                points = [point.float() for point in points]
                feats, coords, sizes = self.voxelize(points)
                batch_size = coords[-1, 0] + 1
        else:
            # NOTE(knzo25): onnx inference. Voxelization happens outside the graph
            with torch.cuda.amp.autocast(enabled=False):
                # with torch.autocast('cuda', enabled=False):
                feats = batch_inputs_dict["voxels"]["voxels"]
                coords = batch_inputs_dict["voxels"]["coors"]
                sizes = batch_inputs_dict["voxels"]["num_points_per_voxel"]

                # NOTE(knzo25): onnx demmands this
                # batch_size = coords[-1, 0] + 1
                batch_size = 1

                assert self.voxelize_reduce
                if self.voxelize_reduce:
                    feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)

        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(
        self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs
    ) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, _ = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get("imgs", None)# torch.Size([1, 5, 3, 384, 704])
        points = batch_inputs_dict.get("points", None)# points[0].shape torch.Size([170579, 3])
        features = []
        depth_loss = 0.0

        if imgs is not None and "lidar2img" not in batch_inputs_dict:
            # NOTE(knzo25): normal training and testing
            imgs = imgs.contiguous() # [B, N_cam, 3, H, W]
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            # 读取每个 sample 的标定矩阵
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta["lidar2img"])
                camera_intrinsics.append(meta["cam2img"])
                camera2lidar.append(meta["cam2lidar"])
                img_aug_matrix.append(meta.get("img_aug_matrix", np.eye(4)))
                lidar_aug_matrix.append(meta.get("lidar_aug_matrix", np.eye(4)))
            # 转成 tensor
            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            #  提取图像 BEV 特征
            img_feature, depth_loss = self.extract_img_feat(
                imgs,
                deepcopy(points),
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
            )
            features.append(img_feature)
        elif imgs is not None:
            # NOTE(knzo25): onnx inference
            lidar2image = batch_inputs_dict["lidar2img"]
            camera_intrinsics = batch_inputs_dict["cam2img"]
            camera2lidar = batch_inputs_dict["cam2lidar"]
            img_aug_matrix = batch_inputs_dict["img_aug_matrix"]
            lidar_aug_matrix = batch_inputs_dict["lidar_aug_matrix"]

            # NOTE(knzo25): originally BEVFusion uses all the points
            # which could be a bit slow. For now I am using only
            # the centroids, which is also suboptimal, but using
            # all the voxels produce errors in TensorRT,
            # so this will be fixed for the next version
            # (ScatterElements bug, or simply null voxels break the equation)
            feats = batch_inputs_dict["voxels"]["voxels"]
            sizes = batch_inputs_dict["voxels"]["num_points_per_voxel"]

            feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)

            geom_feats = batch_inputs_dict["geom_feats"]
            img_feature, depth_loss = self.extract_img_feat(
                imgs,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batch_input_metas,
                geom_feats=geom_feats,
            )
            features.append(img_feature)

        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        return x, depth_loss

    def loss(
        self, batch_inputs_dict: Dict[str, Optional[Tensor]], batch_data_samples: List[Det3DDataSample], **kwargs
    ) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        losses = dict()
        if (self.view_transform):                
            feats, depth_loss = self.extract_feat(batch_inputs_dict, batch_input_metas)
            # losses["depth_loss"] = depth_loss / 5
        else:
            feats, _ = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        return losses
