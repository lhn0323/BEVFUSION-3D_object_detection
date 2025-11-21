# modify from https://github.com/mit-han-lab/bevfusion
from typing import Tuple

import torch
from mmdet3d.registry import MODELS
from torch import nn

from .ops import bev_pool


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


class BaseViewTransform(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self): #创建视锥空间视，锥体（frustum）通常用于表示从相机视角看到的三维空间
        iH, iW = self.image_size  # 图像的高度和宽度 384 704
        fH, fW = self.feature_size  # 特征图的高度和宽度 48 88
        
        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW) # View=reshape -1是自动计算 torch.Size([118, 48, 88])
        #创建一个深度的张量 ds，它是从 self.dbound [1.0, 60.0, 0.5]范围内均匀生成的深度值，并将其扩展到特征图的宽高维度 fH x fW
        D, _, _ = ds.shape

        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW) 
        #linspace生成一个长度为 fW 的 1D 数组 将像素坐标 0 ~ 703 平均分成 88 份，生成 88 个横坐标
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        frustum = torch.stack((xs, ys, ds), -1)  #构成一个形状为 (D, fH, fW, 3) 的张量，表示每个深度值对应的 3D 坐标（x, y, z）
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,      # shape (B, N, 3, 3)
        camera2lidar_trans,     # shape (B, N, 3)
        intrins_inverse,        # shape (B, N, 3, 3) inverse intrinsics
        post_rots_inverse,      # shape (B, N, 3, 3) inverse of image augmentation rotations
        post_trans,             # shape (B, N, 3) image augmentation translations
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape # 1, 5, 3

        # undo post-transformation
        # B x N x D x H x W x 3
        # 逆图像增强 和 逆图像旋转 其中self.frustum D（深度 bins）fH（特征图 height）fW（特征图 width）(x, y, z) 
        # 3D 射线模板 ——用来表示每个像素列和 depth bin 的射线方向，不含任何相机外参
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)# self.frustum.shape torch.Size([118, 48, 88, 3])
        points = post_rots_inverse.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)) # points.shape ([1, 5, 118, 48, 88, 3, 1])
        # cam_to_lidar 透视投影的逆过程 P_cam = (x * z, y * z, z)
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        # 把 camera intrinsics inverse 与 camera->lidar 旋转合并：相当于从像素坐标直接生成到雷达坐标的旋转变换
        combine = camera2lidar_rots.matmul(intrins_inverse) # torch.Size([1, 5, 3, 3])
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # 最后加上 camera2lidar_trans（平移）
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)
        # 如果有 extra_rots和extra_trans（来自 lidar augmentation）则再做一次旋转和平移 逆数据增强
        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)
        # 返回 points：形状 (B, N, D, H, W, 3) ([1, 5, 118, 48, 88, 3, 1])—— 每个像素每个深度 bin 对应的 3D 点在 lidar/world 坐标系中位置
        return points
    

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool_aux(self, geom_feats):

        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W
        assert C == 3

        """ frustrum_numpy = geom_feats.cpu().numpy() """

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [torch.full([Nprime // B, 1], ix, device=geom_feats.device, dtype=torch.long) for ix in range(B)]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )

        geom_feats = geom_feats[kept]

        """ data = {}
        data["frustum"] = frustrum_numpy
        data["kept"] = kept.cpu().numpy()
        import pickle
        with open("frustum.pkl", "wb") as f:
            pickle.dump(data, f) """

        # TODO(knzo25): make this more elegant
        D, H, W = self.nx[2], self.nx[0], self.nx[1]

        ranks = geom_feats[:, 0] * (W * D * B) + geom_feats[:, 1] * (D * B) + geom_feats[:, 2] * B + geom_feats[:, 3]
        indices = ranks.argsort()

        ranks = ranks[indices]
        geom_feats = geom_feats[indices]

        return geom_feats, kept, ranks, indices

    def bev_pool(self, x, geom_feats):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # Taken out of bev_pool for pre-computation
        geom_feats, kept, ranks, indices = self.bev_pool_aux(geom_feats)

        x = x[kept]

        assert x.shape[0] == geom_feats.shape[0]

        x = x[indices]

        """ import pickle
        with open("precomputed_features.pkl", "rb") as f:
            data = pickle.load(f) """

        x = bev_pool(x, geom_feats, ranks, B, self.nx[2], self.nx[0], self.nx[1], self.training)

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def bev_pool_precomputed(self, x, geom_feats, kept, ranks, indices):

        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        x = x[kept]
        assert x.shape[0] == geom_feats.shape[0]

        x = x[indices]
        x = bev_pool(x, geom_feats, ranks, B, self.nx[2], self.nx[0], self.nx[1], self.training)

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(
        self,
        img,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        camera_intrinsics_inverse,
        img_aug_matrix_inverse,
        lidar_aug_matrix_inverse,
        geom_feats_precomputed,
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        if geom_feats_precomputed is not None:
            geom_feats, kept, ranks, indices = geom_feats_precomputed
            x = self.get_cam_feats(img)
            x = self.bev_pool_precomputed(x, geom_feats, kept, ranks, indices)

        else:

            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                torch.inverse(intrins),
                torch.inverse(post_rots),
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )
            # depth is not connected to the calibration
            # on_img is
            # is also flattened_indices
            x = self.get_cam_feats(img)
            x = self.bev_pool(x, geom)

        return x


@MODELS.register_module()
class LSSTransform(BaseViewTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x


class BaseDepthTransform(BaseViewTransform):

    def forward(
        self,
        img,
        points,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        camera_intrinsics_inverse,
        img_aug_matrix_inverse,
        lidar_aug_matrix_inverse,
        geom_feats_precomputed,
    ):
        # 从增强矩阵中提取变换矩阵（用于后面将 points 投回 image）
        post_trans = img_aug_matrix[..., :3, 3]# torch.Size([1, 5, 3])
        camera2lidar_rots = camera2lidar[..., :3, :3]# torch.Size([1, 5, 3, 3])
        camera2lidar_trans = camera2lidar[..., :3, 3] # torch.Size([1, 5, 3]) 

        if lidar_aug_matrix_inverse is None:
            lidar_aug_matrix_inverse = torch.inverse(lidar_aug_matrix) # torch.Size([1, 4, 4])
        #LiDAR 原始点云数据,points 的关键作用：生成稀疏的真值深度图 (depth)，用于监督模型的深度预测
        batch_size = len(points) # points[0].shape torch.Size([178884, 3]) batch_size=1
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device) 
        # torch.Size([1, 5, 1, 384, 704]) 初始化一个张量 depth 来保存预测的深度值，在1这个维度体现，张量的大小基于批量大小、图像的高宽和深度通道。
        
        # 遍历每个批次中的样本，将 LiDAR 点投影到每个相机图像平面，构建稀疏的真值深度图 depth
        for b in range(batch_size):   
            # P_lidar → 逆增强 → 相机坐标系 → 透视变换 → 图像增强 → (row, col)
            cur_coords = points[b][:, :3]    # 提取点云3D 坐标 (x, y, z).(num_points_b, 3) torch.Size([176662, 3])
            cur_img_aug_matrix = img_aug_matrix[b] # (5,4,4)
            cur_lidar_aug_matrix = lidar_aug_matrix[b] # (4,4)
            cur_lidar2image = lidar2image[b] # (5,4,4)

            # inverse aug  激光雷达点的逆变换，激光雷达点 cur_coords 先减去激光雷达增强矩阵的平移部分，然后进行旋转变换，使用的是 lidar_aug_matrix_inverse
            # 1)逆增强：把点从增强后的坐标还原到原始 LiDAR 坐标系
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = lidar_aug_matrix_inverse[b, :3, :3].matmul(cur_coords.transpose(1, 0))
           
            # 2) 转到相机坐标系 (每个相机)
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
           
            # 3) 透视除法得到像素坐标 (x/z, y/z)。先提取深度 z
            dist = cur_coords[:, 2, :]  #提取深度值 [:, 2, :] 提取所有批次、第2个通道（z坐标/深度值）、所有点。提取变换后点的 Z 坐标，这正是点到相机平面的距离，即深度值
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5) # 并对深度值进行裁剪，避免除以零的情况
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] #将 3D 坐标 (X, Y, Z) 投影到 2D 齐次坐标 (X/Z, Y/Z, 1).透视除法 cur_coords[:, :2, :]：所有批次、前2个通道（x, y坐标）、所有点；cur_coords[:, 2:3, :]：所有批次、第2个通道（深度z）、所有点（保持维度）；执行除法：(x, y) / z → 完成3D到2D的投影

            # 4) 应用 image augmentation（仿射）使坐标与输入 img 形变一致 进行图像增强。将 2D 投影坐标应用图像增强矩阵，使坐标与输入图像 img 的增强状态保持一致
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2) 
            # cur_coords shape = [5, 176662, 2] 每个相机各自拥有一份投影结果

            # normalize coords for grid sample 保证 scatter depth 时深度写入正确的像素位置
            # 5) 交换坐标顺序以匹配 grid / image indexing（row, col） “数学坐标系下的 (x, y)”转换成“图像像素空间的 (row=y, col=x)”
            cur_coords = cur_coords[..., [1, 0]]
            # 6) 判断哪些投影点落在图像中
            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            ) #判断变换后的 2D 点是否落在图像边界内

            # NOTE(knzo25): in the original code, a per-image loop was
            # implemented to compute the depth. However, it fixes the number
            # of images, which is not desired for deployment (the number
            # of images may change due to frame drops).
            # For this reason, I modified the code to use tensor operations,
            # but the results will change due to indexing having potential
            # duplicates !. In practce, only about 0.01% of the elements will
            # have different results...
            #在原始代码中，通过针对每帧图像的循环来计算深度。但该方案固定了图像数量，这不符合部署需求（因帧丢失可能导致图像数量变化）。
            #为此改用张量运算实现，但索引可能存在重复导致结果变化！实际中仅约0.01%的元素会产生差异结果...对落在图像中的点，将其对应的深度值更新到 depth 张量中

            # 7) 把 3D LiDAR 点投影到每个相机的二维图像上，并把该点的深度写进一张稀疏的深度图 depth[B, N, 1, H, W] 稀疏深度图=用于监督 DepthNet 的真值深度
            #   1.找出哪些投影点落在图像内 提取有效投影点的相机索引和点云索引
            #     on_img([5, 176662])的true or false是一个布尔掩码，标记哪些3D点投影在图像范围内，第 p 个 LiDAR 点投影到第几个相机后落在图像内
            #    [F,F,T,T,T,F,...],   # cam0: 哪些点落在 cam0 视野内 [F,F,F,T,F,T,...],   # cam1  
            indices = torch.nonzero(on_img, as_tuple=False) 
            #   2,分离相机编号与点编号 indices是[[camera_id, point_id],[camera_id, point_id]] ([34823, 2])
            camera_indices = indices[:, 0] # camera_indices	这个点来自哪个 camera ([34823])0-4
            point_indices = indices[:, 1] # point_indices	这个点是 LiDAR 点列表中的哪个 index ([34823])
            #   3.取出投影后的像素坐标 (row, col),取出对应的深度 dist
            masked_coords = cur_coords[camera_indices, point_indices].long() 
            #   4.取出对应的深度 dist 之前拿到的z坐标
            masked_dist = dist[camera_indices, point_indices] # torch.Size([74631, 2])
            depth = depth.to(masked_dist.dtype) # ([1, 5, 1, 384, 704]) 
            batch_size, num_imgs, channels, height, width = depth.shape
            # Depth tensor should have only one channel in this implementation
            assert channels == 1
            #    6.深度图 flat 化（便于 scatter）
            depth_flat = depth.view(batch_size, num_imgs, channels, -1)  #将深度图展平，便于后续的scatter操作 torch.Size([1, 5, 1, 270336])
            #    7.为每个投影点计算平铺索引，把 (camera, row, col) 转成一维索引
            #    对于每个投影点：索引 = 相机索引 × (图像高度 × 图像宽度) + 行坐标 × 宽度 + 列坐标。等价于把每张图平铺在一起，先放 camera0 的 HW 元素，再放 camera1 的 HW 元素
            flattened_indices = camera_indices * height * width + masked_coords[:, 0] * width + masked_coords[:, 1] # torch.Size([74631])
            #    8.将有效的 LiDAR 深度值 (masked_dist) 写入全零的扁平化更新张量 depth 中，从而生成了 稀疏的真值深度图
            updates_flat = torch.zeros((num_imgs * channels * height * width), device=depth.device)
            #     scatter把 masked_dist 中的每个深度值写入到 updates_flat 的 flat_index 对应的位置。dim=0：在第0维度进行scatter操作
            updates_flat.scatter_(dim=0, index=flattened_indices, src=masked_dist)
            #     9.将更新后的扁平张量恢复为原始形状batch
            depth_flat[b] = updates_flat.view(num_imgs, channels, height * width)
            depth = depth_flat.view(batch_size, num_imgs, channels, height, width) 
            # 此时的depth便为稀疏的真值深度图 torch.Size([1, 5, 1, 384, 704])（非每像素都有值）

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        if geom_feats_precomputed is not None:
            # In inference, the geom_feats are precomputed 
            # 推理时的快速路径：如果外部已经预计算好了 geom_feats（映射）
            # geom_feats_precomputed 解包：包含用于 bev pooling 的预计算数据
            geom_feats, kept, ranks, indices, camera_mask = geom_feats_precomputed
            # 从图像特征与稀疏 depth 生成 per-camera 的 lifted features
            x, est_depth_distr, gt_depth_distr, counts_3d = self.get_cam_feats(img, depth)

            """ data = {}
            data["img"] = img.cpu()
            data["depth"] = depth.cpu()
            data["x"] = x.cpu()
            import pickle
            with open("depth_deploy.pkl", "wb") as f:
                pickle.dump(data, f) """

            # At inference, if a camera is missing, we just mask the features
            # example: camera_mask = [1, 1, 1, 0, 1, 1]
            # camera_mask: 指明哪些相机在此帧有效（1）或缺失（0），用于部署时相机丢帧处理
            camera_mask = camera_mask.view(1, -1, 1, 1, 1, 1)  # camera_mask.shape = [1, 6, 1, 1, 1, 1]
            # 把被 mask 掉的相机的 features 清零，再用预计算的 pooling 快速汇聚到 BEV
            x = self.bev_pool_precomputed(x * camera_mask, geom_feats, kept, ranks, indices)
        else:
            # 动态计算 geometry
            intrins_inverse = torch.inverse(cam_intrinsic)[..., :3, :3] # torch.Size([1, 5, 3, 3])
            post_rots_inverse = torch.inverse(img_aug_matrix)[..., :3, :3] # post_rots_inverse
             # get_geometry 计算 frustum（像素 × depth_bin）到世界/雷达坐标的点位置映射
            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                intrins_inverse,
                post_rots_inverse,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            )

            # Load from the pkl
            """ import pickle
            with open("precomputed_features.pkl", "rb") as f:
                data = pickle.load(f) """
            # 从图像特征与稀疏 depth 生成 per-camera 的 lifted features 为每个像素找到所属的特征图 cell
            x, est_depth_distr, gt_depth_distr, counts_3d = self.get_cam_feats(img, depth)

            """ import pickle
            with open("depth_deploy.pkl", "rb") as f:
                data = pickle.load(f) """
            # 根据动态计算得到的 geom 做 BEV pooling（把 3D volume 特征汇聚到 BEV 网格）
            x = self.bev_pool(x, geom)

        if self.training:
            # depth loss: 训练时，使用稀疏 LiDAR 生成的 gt_depth_distr 对 est_depth_distr 做交叉熵监督
            """counts_3d_aux = counts_3d.permute(0,1,4,2,3).unsqueeze(-1)
            gt_feats = gt_depth_distr.permute(0,1,4,2,3).unsqueeze(-1) * (counts_3d_aux > 0).float()
            est_feats = est_depth_distr.permute(0,1,4,2,3).unsqueeze(-1)

            num_cameras = gt_feats.shape[1]

            gt_bev_feats = self.bev_pool_precomputed(gt_feats, geom_feats, kept, ranks, indices)
            est_bev_feats = self.bev_pool_precomputed(est_feats, geom_feats, kept, ranks, indices)

            import pickle
            data = {}
            data["gt_bev_feats"] = gt_bev_feats.cpu().numpy()
            data["est_bev_feats"] = est_bev_feats.cpu().numpy()

            for i in range(num_cameras):
                gt_feats_aux = torch.zeros_like(gt_feats)
                gt_feats_aux[:,i] = gt_feats[:,i]
                gt_bev_feats_aux = self.bev_pool_precomputed(gt_feats_aux, geom_feats, kept, ranks, indices)

                est_feats_aux = torch.zeros_like(est_feats)
                est_feats_aux[:,i] = est_feats[:,i]
                est_bev_feats_aux = self.bev_pool_precomputed(est_feats_aux, geom_feats, kept, ranks, indices)

                data[f"gt_bev_feats_{i}"] = gt_bev_feats_aux.cpu().numpy()
                data[f"est_bev_feats_{i}"] = est_bev_feats_aux.cpu().numpy()

            with open("bev_features.pkl", "wb") as f:
                pickle.dump(data, f)"""

            mask_flat = counts_3d.sum(dim=-1).view(-1) > 0

            gt_depth_distr_flat = gt_depth_distr.view(-1, self.D)
            est_depth_distr_flat = est_depth_distr.reshape(-1, self.D)

            cross_ent = -torch.sum(gt_depth_distr_flat * torch.log(est_depth_distr_flat + 1e-8), dim=-1)
            cross_ent_masked = cross_ent * mask_flat.float()
            depth_loss = torch.sum(cross_ent_masked) / (mask_flat.sum() + 1e-8)
        else:
            depth_loss = 0.0

        return x, depth_loss


@MODELS.register_module()
class DepthLSSTransform(BaseDepthTransform):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_cam_feats(self, x, d):
        # x: image features from img_neck, shape (B, N, C, fH, fW)
        # d: sparse depth map (from LiDAR), shape (B, N, 1, H, W) torch.Size([5, 1, 384, 704])
        # 把稀疏深度图 H×W 原图规模 对齐到 feature map fH×fW (下采样8倍)
        B, N, C, fH, fW = x.shape # 1,5,256,48,88
        h, w = self.image_size #384, 704
        BN = B * N # 5
        #  将深度 & 图像特征展平成 BN 的 batch（把相机维合并）把多相机视为独立样本处理
        d = d.view(BN, *d.shape[2:]) # depth是稀疏真值深度图 torch.Size([5, 1, 384, 704])
        x = x.view(BN, C, fH, fW) # image features torch.Size([5, 256, 48, 88])

        # =================== TEST
        if self.training or True:
            # 1)为每个像素找到所属的特征图 cell
            #   1.构造 pixel 坐标网格
            camera_id = torch.arange(BN).view(-1, 1, 1).expand(BN, h, w) # Size([5, 384, 704])
            rows = torch.arange(h).view(1, -1, 1).expand(BN, h, w)# Size([5, 384, 704])
            cols = torch.arange(w).view(1, 1, -1).expand(BN, h, w)# Size([5, 384, 704])
            #   2.计算像素落在哪个特征图 cell
            cell_j = rows // (h // fH) # ([5, 384, 704]) // 8  feature map 的 row index Size([5, 384, 704])
            cell_i = cols // (w // fW)
            #   3.计算 cell 的 “flat id” 第 BN 个相机的第 (cell_j,cell_i) 个像素对应的 3D voxel cell 表示像素对应落入哪个 特征图 cell(j,i)
            cell_id = camera_id * fH * fW + cell_j * fW + cell_i
            cell_id = cell_id.to(device=d.device)

            # 2)拿到每个像素点（在每个相机）都会得到一个对应的 bin index 表示深度 d 落入哪个 3D 深度层（bin）
            #   把深度离散成 D 个 bin (d+0.25−1.0) / 0.5 = 118 
            #   self.dbound是[1.0, 60.0, 0.5] clamp:深度 d 被限制到 [1.0, 59.75] 之间 将深度区间从 [1, 60] 映射到 [0, 59]
            #   bin_index = floor((d + d_step/2 - d_min) / d_step)对 d 做向下取整但居中对齐 
            #   相比于bin_index = floor((d - d_min) / d_step)这种方式符号更稳定，对稀疏深度监督更可靠
            dist_bins = (
                d.clamp(min=self.dbound[0], max=self.dbound[1] - 0.5 * self.dbound[2])
                + 0.5 * self.dbound[2]
                - self.dbound[0]
            ) / self.dbound[2]
            dist_bins = dist_bins.long() # torch.Size([5, 1, 384, 704])

            # 3)构造 flat index:这个像素属于第 flat_cell_id 个 cell 和 cell 下的第 dist_bin 个深度层
            # flat_index 是一个唯一的一维编号用于表示 3D 空间中的一个体素 Voxel：feature cell (fH × fW) × depth bin (D) 对应的 3D 网格 cell（一个小方块）
            flat_cell_id = cell_id.view(-1) # 是每个像素(每个相机)对应的 feature id torch.Size([1351680])
            flat_dist_bin = dist_bins.view(-1)# 每个像素(每个相机)对应的深度 torch.Size([1351680])
            flat_index = flat_cell_id * self.D + flat_dist_bin

            counts_flat = torch.zeros(BN * fH * fW * self.D, dtype=torch.float, device=d.device)
            counts_flat.scatter_add_(
                0, flat_index, torch.ones_like(flat_index, dtype=torch.float, device=flat_index.device)
            )

            counts_3d = counts_flat.view(B, N, fH, fW, self.D)
            counts_3d[..., 0] = 0.0

            # mask_flat = counts_3d.sum(dim=-1).view(-1) > 0

            # gt_depth_distr = torch.softmax(counts_3d, dim=-1)
            gt_depth_distr = counts_3d / (counts_3d.sum(dim=-1, keepdim=True) + 1e-8)
            # gt_depth_distr_flat = gt_depth_distr.view(-1, self.D)
            # =================== TEST
        else:
            gt_depth_distr = None
            counts_3d = None

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        est_depth_distr = depth.permute(0, 2, 3, 1).reshape(B, N, fH, fW, self.D)

        if self.training:
            depth_aux = gt_depth_distr.view(B * N, fH, fW, self.D).permute(0, 3, 1, 2)
            depth = depth + (torch.maximum(depth_aux, depth) - depth).detach()
        # Need to match the (B, N, H, W, D) order

        # est_depth_distr_flat = est_depth_distr.reshape(-1, self.D)

        """ import pickle
        data = {}
        data["gt_depth"] = gt_depth_distr.cpu().numpy()
        data["estimated_depth"] = est_depth_distr.cpu().numpy()
        data["counts"] = counts_3d.cpu().numpy()
        with open("estimated_depth.pkl", "wb") as f:
            pickle.dump(data, f) """

        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, est_depth_distr, gt_depth_distr, counts_3d

    def forward(self, *args, **kwargs):
        # 调用父类的 forward（BaseDepthTransform.forward），得到 x_bev 和 depth_loss
        x, depth_loss = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x, depth_loss
