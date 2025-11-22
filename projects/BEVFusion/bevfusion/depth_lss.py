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
        iH, iW = self.image_size  # 图像的高度和宽度
        fH, fW = self.feature_size  # 特征图的高度和宽度

        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW) #创建一个深度的张量 ds，它是从 self.dbound 范围内均匀生成的深度值，并将其扩展到特征图的宽高维度 fH x fW
        D, _, _ = ds.shape

        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  #x 坐标：在 $[0, iW-1]$ 范围内均匀采样的 $fW$ 个像素列索引，并扩展到 $D$ 和 $fH$ 维度
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        frustum = torch.stack((xs, ys, ds), -1)  #构成一个形状为 (D, fH, fW, 3) 的张量，表示每个深度值对应的 3D 坐标（x, y, z）
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins_inverse,
        post_rots_inverse,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = post_rots_inverse.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(intrins_inverse)
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

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
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3] #从增强矩阵和相机到激光雷达的变换矩阵中提取出旋转和平移部分，用于后续的变换操作。

        if lidar_aug_matrix_inverse is None:
            lidar_aug_matrix_inverse = torch.inverse(lidar_aug_matrix)

        #LiDAR 原始点云数据,points 的关键作用：生成稀疏的真值深度图 (depth)，用于监督模型的深度预测
        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device)  #根初始化一个张量 depth 来保存预测的深度值，在1这个维度体现，张量的大小基于批量大小、图像的高宽和深度通道。
        for b in range(batch_size):   #这个循环对每个批次中的样本进行处理，提取激光雷达点和变换矩阵,将 LiDAR 点投影到 2D 图像平面
            cur_coords = points[b][:, :3]    #提取的是 3D 坐标 (x, y, z).points其形状通常是 [B, N, num_points_i, 4]，其中 B 是批次大小，N 是相机数量（例如 5 个），num_points_i 是第 i 个相机视锥体内的点数，4 是坐标维度（x, y, z, 强度/其他特征）。但在 BaseDepthTransform 中，通常是按批次和相机分开处理的点列表。
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug  激光雷达点的逆变换，激光雷达点 cur_coords 先减去激光雷达增强矩阵的平移部分，然后进行旋转变换，使用的是 lidar_aug_matrix_inverse
            #应用了逆增强矩阵，目的是将这些点从增强后的坐标系（Augmented Coordinates）还原到原始 LiDAR 坐标系，以匹配未增强的相机内外参。
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = lidar_aug_matrix_inverse[b, :3, :3].matmul(cur_coords.transpose(1, 0))
            # lidar2image   #将 3D 点从 LiDAR 坐标系变换到相机坐标系，使用了 lidar2image 矩阵进行旋转和平移变换
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords  将三维点转换为二维图像坐标。齐次坐标的透视除法
            dist = cur_coords[:, 2, :]  #提取深度值 [:, 2, :] 提取所有批次、第2个通道（z坐标/深度值）、所有点。提取变换后点的 Z 坐标，这正是点到相机平面的距离，即深度值
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5) #并对深度值进行裁剪，避免除以零的情况
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] #将 3D 坐标 (X, Y, Z) 投影到 2D 齐次坐标 (X/Z, Y/Z, 1).透视除法 cur_coords[:, :2, :]：所有批次、前2个通道（x, y坐标）、所有点；cur_coords[:, 2:3, :]：所有批次、第2个通道（深度z）、所有点（保持维度）；执行除法：(x, y) / z → 完成3D到2D的投影

            # imgaug 对坐标进行图像增强操作，再次变换坐标。将 2D 投影坐标应用图像增强矩阵，使坐标与输入图像 img 的增强状态保持一致
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample 坐标归一化以进行网格采样，交换 x 和 y 坐标，可能是为了匹配图像格式（高×宽），3D点在2D画面的位置
            cur_coords = cur_coords[..., [1, 0]]

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
            #在原始代码中，通过针对每帧图像的循环来计算深度。但该方案固定了图像数量，这不符合部署需求（因帧丢失可能导致图像数量变化）。为此我改用张量运算实现，但索引可能存在重复导致结果变化！实际中仅约0.01%的元素会产生差异结果...

#对落在图像中的点，将其对应的深度值更新到 depth 张量中
            indices = torch.nonzero(on_img, as_tuple=False)  #on_img 是一个布尔掩码，标记哪些3D点投影在图像范围内，提取有效投影点的相机索引和点云索引
            camera_indices = indices[:, 0]
            point_indices = indices[:, 1]

            masked_coords = cur_coords[camera_indices, point_indices].long() #投影到图像平面的2D坐标 (x, y)
            masked_dist = dist[camera_indices, point_indices] #对应的深度值（距离相机的距离）
            depth = depth.to(masked_dist.dtype)
            batch_size, num_imgs, channels, height, width = depth.shape
            # Depth tensor should have only one channel in this implementation
            assert channels == 1

            depth_flat = depth.view(batch_size, num_imgs, channels, -1)  #将深度图展平，便于后续的scatter操作
            #计算扁平化索引，对于每个投影点：索引 = 相机索引 × (图像高度 × 图像宽度) + 行坐标 × 宽度 + 列坐标。这相当于将多张图像在内存中连续排列，计算每个投影点的绝对位置
            flattened_indices = camera_indices * height * width + masked_coords[:, 0] * width + masked_coords[:, 1]
            #创建全零的扁平化更新张量；使用 scatter_ 将深度值写入对应位置；dim=0：在第0维度进行scatter操作
            #将有效的 LiDAR 深度值 (masked_dist) 写入了一个零初始化的张量 depth 中，从而生成了 稀疏的真值深度图
            updates_flat = torch.zeros((num_imgs * channels * height * width), device=depth.device)
            updates_flat.scatter_(dim=0, index=flattened_indices, src=masked_dist)
            #将更新后的扁平张量恢复为原始形状
            depth_flat[b] = updates_flat.view(num_imgs, channels, height * width)
            depth = depth_flat.view(batch_size, num_imgs, channels, height, width) #此时的depth便为稀疏的真值深度图

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        if geom_feats_precomputed is not None:
            # In inference, the geom_feats are precomputed 推理走这个
            geom_feats, kept, ranks, indices, camera_mask = geom_feats_precomputed
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
            camera_mask = camera_mask.view(1, -1, 1, 1, 1, 1)  # camera_mask.shape = [1, 6, 1, 1, 1, 1]

            x = self.bev_pool_precomputed(x * camera_mask, geom_feats, kept, ranks, indices)
        else:
            intrins_inverse = torch.inverse(cam_intrinsic)[..., :3, :3]
            post_rots_inverse = torch.inverse(img_aug_matrix)[..., :3, :3]

            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                intrins_inverse,
                post_rots_inverse,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            ) #最终返回的 points (即 geom) 就是 $B \times N \times D \times H \times W \times 3$ 的张量，存储了所有视锥体网格点在 3D 空间中的坐标。

            # Load from the pkl
            """ import pickle
            with open("precomputed_features.pkl", "rb") as f:
                data = pickle.load(f) """

            x, est_depth_distr, gt_depth_distr, counts_3d = self.get_cam_feats(img, depth)

            """ import pickle
            with open("depth_deploy.pkl", "rb") as f:
                data = pickle.load(f) """

            x = self.bev_pool(x, geom)

        if self.training:
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
        B, N, C, fH, fW = x.shape
        h, w = self.image_size
        BN = B * N

        d = d.view(BN, *d.shape[2:])
        x = x.view(BN, C, fH, fW)

        # =================== TEST
        if self.training or True:
            camera_id = torch.arange(BN).view(-1, 1, 1).expand(BN, h, w)
            rows = torch.arange(h).view(1, -1, 1).expand(BN, h, w)
            cols = torch.arange(w).view(1, 1, -1).expand(BN, h, w)

            cell_j = rows // (h // fH)
            cell_i = cols // (w // fW)

            cell_id = camera_id * fH * fW + cell_j * fW + cell_i
            cell_id = cell_id.to(device=d.device)

            dist_bins = (
                d.clamp(min=self.dbound[0], max=self.dbound[1] - 0.5 * self.dbound[2])
                + 0.5 * self.dbound[2]
                - self.dbound[0]
            ) / self.dbound[2]
            dist_bins = dist_bins.long()

            flat_cell_id = cell_id.view(-1)
            flat_dist_bin = dist_bins.view(-1)

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
        x, depth_loss = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x, depth_loss
