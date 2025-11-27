# modify from https://github.com/mit-han-lab/bevfusion
from typing import Tuple

import torch
from mmdet3d.registry import MODELS
from torch import nn

from .ops import bev_pool


# dx - 体素尺寸 每个体素在x、y、z方向上的物理尺寸（米）
#bx - 起始偏移 第一个体素中心点的坐标，体素网格从边界最小值开始，但坐标指向体素中心
#nx - 体素数量 每个维度上的体素数量
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
        self.xbound = xbound      #-54.0：X轴最小值（米）54.0：X轴最大值（米）0.3：X轴体素大小（米）总范围：108米，体素数量：108 / 0.3 = 360
        self.ybound = ybound
        self.zbound = zbound  #-10.0：Z轴最小值（米）10.0：Z轴最大值（米）20.0：Z轴体素大小（米）总范围：20米，体素数量：20 / 20 = 1
        self.dbound = dbound   #1.0：最近距离（米）60.0：最远距离（米）0.5：深度分辨率（米）总范围：59米，深度bin数量：(60-1) / 0.5 = 118

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def create_frustum(self): #创建视锥空间视，锥体（frustum）通常用于表示从相机视角看到的三维空间
        iH, iW = self.image_size  #384,704 图像的高度和宽度
        fH, fW = self.feature_size  #48,88 特征图的高度和宽度
        #self.dbound [1.0, 60.0, 0.5] torch.arange 使用解包运算符 * 接收 self.dbound 的三个参数：start=1.0, end=60.0, step=0.5 。生成118个深度值
        #view(-1, 1, 1): 将一维深度序列重塑为 $\{[D, 1, 1]}$ 的形状（即 ${[118, 1, 1]}$）。expand(-1, fH, fW): 沿着后两个维度进行广播 (Broadcast)。fH 是特征图高度 $48$。fW 是特征图宽度 $88$。ds 的最终形状为 $\{[D, fH, fW]}$（即 $\{[118, 48, 88]}$
        #对于特征图上的每个像素 $(i, j)$，ds[:, i, j] 都包含一个形状为 $[D]$ 的向量，该向量列出了 $118$ 个离散深度值，序列中的每个数值 $d_i$（例如 $1.0, 1.5, 2.0, ..., 59.5$）代表第 $i$ 个深度 Bin 的中心距离
        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  #[118, 48, 88]
        D, _, _ = ds.shape
        #这 $fW$ 个点代表了特征图 $fW$ 维度上的每个特征单元格对应的原始图像上的 X 轴中心像素坐标。例如，如果原始图像 $iW=704$ 且 $fW=88$，则每隔 8 个像素采样一个点，得到的 x 坐标序列为 [0, 8, 16, ..., 696]
        #xs[k, i, j] 的值，代表第 $k$ 个深度 Bin、第 $i$ 行、第 $j$ 列的视锥体单元格所对应的 图像 X 轴坐标
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  #torch.Size([118, 48, 88])#x 坐标：在 $[0, iW-1]$ 范围内均匀采样的 fW 个像素列索引，并扩展到D和fH维度
        #ys[k, i, j] 的值，代表第 $k$ 个深度 Bin、第 $i$ 行、第 $j$ 列的视锥体单元格所对应的 图像 Y 轴坐标
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW) #torch.Size([118, 48, 88])#y 坐标：在 $[0, iH-1]$ 范围内均匀采样的 $fH$ 个像素行索引，并扩展到 $D$ 和 $fW$ 维度。
        #frustum[k, i, j] 的值，代表第 $k$ 个深度 Bin、第 $i$ 行、第 $j$ 列的视锥体单元格所对应的 图像 X,Y,D 轴坐标
        frustum = torch.stack((xs, ys, ds), -1)  #torch.Size([118, 48, 88,3])#构成一个形状为 (D, fH, fW, 3) 的张量，表示每个深度值对应的 3D 坐标（x, y, z），通过堆叠 $(x, y, d)$ 构成 3D 采样点集
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
        B, N, _ = camera2lidar_trans.shape #torch.Size([1, 5, 3])

        # undo post-transformation
        # B x N x D x H x W x 3
        #将视锥体网格点 (self.frustum) 应用逆图像增强/后处理（如裁剪、缩放的逆操作），将点从增强后的图像空间还原到原始图像空间
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3) #torch.Size([1, 5, 118, 48, 88, 3]) 平移变换
        points = post_rots_inverse.view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)) #torch.Size([1, 5, 118, 48, 88, 3, 1]) 旋转变换 points.unsqueeze(-1)指
        # cam_to_lidar
        #实现了逆透视操作。在 self.frustum 中，点的坐标 $(u', v', d)$ 代表归一化坐标 $(u'/d, v'/d, 1)$。现在通过乘以深度 $d$ 恢复到 3D 齐次坐标系下的坐标 $(u', v', d)$。
        #2D 像素坐标系到 3D 视锥体坐标系的关键变换
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        ) 
        #相机到 LiDAR 变换，这一行执行了关键的逆投影，并结合了相机外参 
        combine = camera2lidar_rots.matmul(intrins_inverse)
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1) #3D 点从相机坐标系转换到了 LiDAR 坐标系的旋转部分
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3) #加上相机到 LiDAR 的平移向量，完成完整的 3D 坐标变换

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

        return points  #被精确定位到 3D LiDAR/BEV 空间中

    def get_cam_feats(self, x):
        raise NotImplementedError

    def bev_pool_aux(self, geom_feats):

        B, N, D, H, W, C = geom_feats.shape
        Nprime = B * N * D * H * W
        assert C == 3

        """ frustrum_numpy = geom_feats.cpu().numpy() """

        # flatten indices
        #将连续的 3D 坐标 $(\mathbf{x}_{\text{cont}})$ 转换为 BEV 网格索引 $(\mathbf{x}_{\text{idx}})$
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long() #self.bx 是 BEV 边界，self.dx 是 BEV 网格分辨率。此公式将坐标转换为 BEV 网格的 $(X, Y, Z)$ 索引
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [torch.full([Nprime // B, 1], ix, device=geom_feats.device, dtype=torch.long) for ix in range(B)]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1) #为每个点添加其所属的批次索引 B，使聚合操作能够区分不同样本

        # filter out points that are outside box 
        #移除所有索引超出预定义 BEV 边界 (self.nx) 的特征点。kept 是一个布尔掩码，用于后续筛选特征 x
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
        #为每个有效的 3D 网格单元分配一个唯一的一维索引 (Rank)，以保证同一单元内的所有特征点在排序后是相邻的。使用 Z 轴、Y 轴、X 轴和 Batch 轴的索引加权求和，得到一个唯一的整数 ranks。
        ranks = geom_feats[:, 0] * (W * D * B) + geom_feats[:, 1] * (D * B) + geom_feats[:, 2] * B + geom_feats[:, 3]
        #对特征点根据 ranks 进行排序。这是实现高效 BEV 聚合的关键。
        #排序后，所有将要投影到同一 BEV 单元的特征点现在在张量中是连续排列的。
        indices = ranks.argsort()

        ranks = ranks[indices]
        geom_feats = geom_feats[indices]

        return geom_feats, kept, ranks, indices

    #geom_feats  B, N, D, H, W, C,3
    def bev_pool(self, x, geom_feats):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # Taken out of bev_pool for pre-computation  负责将上一步 get_geometry 计算出的连续 3D 坐标，映射到离散的 BEV 网格索引
        geom_feats, kept, ranks, indices = self.bev_pool_aux(geom_feats)

        x = x[kept]  #确保特征和坐标数量一致

        assert x.shape[0] == geom_feats.shape[0]

        x = x[indices]    #使用 indices 对 x 排序，使其与 geom_feats 的顺序一致（同一 BEV 单元的特征相邻）。

        """ import pickle
        with open("precomputed_features.pkl", "rb") as f:
            data = pickle.load(f) """

        x = bev_pool(x, geom_feats, ranks, B, self.nx[2], self.nx[0], self.nx[1], self.training)

        # collapse Z  将聚合后的 3D BEV 特征从 $B \times C \times D \times H_{\text{bev}} \times W_{\text{bev}}$ 转换为最终的 2D BEV 特征。
        #将深度维度 $D$ (Z 轴) 展平并连接到特征维度 $C$ 上，得到最终的 2D 鸟瞰图 (BEV) 特征，形状为 $B \times (C \cdot D) \times H_{\text{bev}} \times W_{\text{bev}}$。
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
            lidar_aug_matrix_inverse = torch.inverse(lidar_aug_matrix) #torch.Size([1, 4, 4])

        #LiDAR 原始点云数据,points 的关键作用：生成稀疏的真值深度图 (depth)，用于监督模型的深度预测
        batch_size = len(points) #torch.Size([178208, 3])
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device)  #torch.Size([1, 5, 1, 384, 704]) #根初始化一个张量 depth 来保存预测的深度值，在1这个维度体现，张量的大小基于批量大小、图像的高宽和深度通道。
        for b in range(batch_size):   #这个循环对每个批次中的样本进行处理，提取激光雷达点和变换矩阵,将 LiDAR 点投影到 2D 图像平面
            cur_coords = points[b][:, :3]    #torch.Size([178208, 3]) #提取的是 3D 坐标 (x, y, z).points其形状通常是 [B, N, num_points_i, 4]，其中 B 是批次大小，N 是相机数量（例如 5 个），num_points_i 是第 i 个相机视锥体内的点数，4 是坐标维度（x, y, z, 强度/其他特征）。但在 BaseDepthTransform 中，通常是按批次和相机分开处理的点列表。
            cur_img_aug_matrix = img_aug_matrix[b] #torch.Size([5, 4, 4])
            cur_lidar_aug_matrix = lidar_aug_matrix[b]#torch.Size([4, 4])
            cur_lidar2image = lidar2image[b]#torch.Size([5, 4, 4])

            # inverse aug  激光雷达点的逆变换，激光雷达点 cur_coords 先减去激光雷达增强矩阵的平移部分，然后进行旋转变换，使用的是 lidar_aug_matrix_inverse
            #应用了逆增强矩阵，目的是将这些点从增强后的坐标系（Augmented Coordinates）还原到原始 LiDAR 坐标系，以匹配未增强的相机内外参。
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = lidar_aug_matrix_inverse[b, :3, :3].matmul(cur_coords.transpose(1, 0))   #[3,3]*[3,178208]=[3,178208]
            # lidar2image   #将 3D 点从 LiDAR 坐标系变换到相机坐标系，使用了 lidar2image 矩阵进行旋转和平移变换
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords) #torch.Size([5, 3, 178208])
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1) #torch.Size([5, 3, 178208])
            # get 2d coords  将三维点转换为二维图像坐标。齐次坐标的透视除法
            dist = cur_coords[:, 2, :]  #torch.Size([5, 178208])#提取深度值 [:, 2, :] 提取所有批次、第2个通道（z坐标/深度值）、所有点。提取变换后点的 Z 坐标，这正是点到相机平面的距离，即深度值
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5) #并对深度值进行裁剪，避免除以零的情况
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] #torch.Size([5, 2, 178208]) 将 3D 坐标 (X, Y, Z) 投影到 2D 齐次坐标 (X/Z, Y/Z, 1).透视除法 cur_coords[:, :2, :]：所有批次、前2个通道（x, y坐标）、所有点；cur_coords[:, 2:3, :]：所有批次、第2个通道（深度z）、所有点（保持维度）；执行除法：(x, y) / z → 完成3D到2D的投影

            # imgaug 对坐标进行图像增强操作，再次变换坐标。将 2D 投影坐标应用图像增强矩阵，使坐标与输入图像 img 的增强状态保持一致
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)  #torch.Size([5, 178208, 2])

            # normalize coords for grid sample 坐标归一化以进行网格采样，交换 x 和 y 坐标，（即 XY\YX 交换）是为了匹配图像格式（高×宽），3D点在2D画面的位置
            cur_coords = cur_coords[..., [1, 0]]  #torch.Size([5, 178208, 2])

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])  #self.image_size[0]=384  cur_coords[..., 0]表示Y [5, 178208]
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1]) #self.image_size[1]=704  cur_coords[..., 1]表示x [5, 178208]
                & (cur_coords[..., 1] >= 0)
            ) #判断变换后的 2D 点是否落在图像边界内 #torch.Size([5, 178208])

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
            indices = torch.nonzero(on_img, as_tuple=False)  #torch.Size([43118, 2])#on_img 是一个布尔掩码，标记哪些3D点投影在图像范围内，indices 返回 $43118$ 行，每行包含两个索引 $(i, j)$，分别表示该有效点在维度 0（相机/批次索引）和维度 1（点云/单元格索引）中的位置
            camera_indices = indices[:, 0]
            point_indices = indices[:, 1]

            masked_coords = cur_coords[camera_indices, point_indices].long() #torch.Size([43118, 2])#投影到图像平面的2D坐标 (x, y) 切片操作,将浮点坐标转换为 64 位整数 使用 camera_indices 和 point_indices 的 配对索引（Pairwise Indexing）来同时从 cur_coords 的前两个维度中选择元素。它选择所有满足 $(i, j)$ 对的坐标，其中 $i$ 来自 camera_indices，$j$ 来自 point_indices
            masked_dist = dist[camera_indices, point_indices] #torch.Size([43118])#对应的深度值（距离相机的距离）
            depth = depth.to(masked_dist.dtype) #程序显式地将 depth 的数据类型设置为与 masked_dist 的数据类型一致
            batch_size, num_imgs, channels, height, width = depth.shape #torch.Size([1, 5, 1, 384, 704])
            # Depth tensor should have only one channel in this implementation
            assert channels == 1

            depth_flat = depth.view(batch_size, num_imgs, channels, -1)  #torch.Size([1, 5, 1, 270336])#将深度图展平，便于后续的scatter操作
            #计算扁平化索引，对于每个投影点：索引 = 相机索引 × (图像高度 × 图像宽度) + 行坐标 × 宽度 + 列坐标。这相当于将多张图像在内存中连续排列，计算每个投影点的绝对位置
            #这一步将每个有效 3D 单元格对应的 2D 像素坐标，转换为在整个批次和所有相机的 2D 图像特征图上的一维索引,用于定位展平后的 2D 图像特征图
            flattened_indices = camera_indices * height * width + masked_coords[:, 0] * width + masked_coords[:, 1] #torch.Size([43118])
            #创建全零的扁平化更新张量；使用 scatter_ 将深度值写入对应位置；dim=0：在第0维度进行scatter操作
            #将有效的 LiDAR 深度值 (masked_dist) 写入了一个零初始化的张量 depth 中，从而生成了 稀疏的真值深度图
            updates_flat = torch.zeros((num_imgs * channels * height * width), device=depth.device)#torch.Size([1351680])
            #index=flattened_indices: 使用上一步计算的一维索引来指定写入位置。src=masked_dist: 使用 masked_dist 中的 浮点深度值 作为要写入的数据
            updates_flat.scatter_(dim=0, index=flattened_indices, src=masked_dist) #scatter_ 是 PyTorch 中将稀疏数据写入密集张量的有效方法
            #将更新后的扁平张量恢复为原始形状
            depth_flat[b] = updates_flat.view(num_imgs, channels, height * width) #torch.Size([5, 1, 270336])
            depth = depth_flat.view(batch_size, num_imgs, channels, height, width) #torch.Size([1, 5, 1, 384, 704])此时的depth便为稀疏的真值深度图 B N DC H W

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
            intrins_inverse = torch.inverse(cam_intrinsic)[..., :3, :3] #torch.Size([1, 5, 3, 3])
            post_rots_inverse = torch.inverse(img_aug_matrix)[..., :3, :3]

            geom = self.get_geometry(
                camera2lidar_rots,
                camera2lidar_trans,
                intrins_inverse,
                post_rots_inverse,
                post_trans,
                extra_rots=extra_rots,
                extra_trans=extra_trans,
            ) #最终返回的 points (即 geom) 就是 $B N  D  H  W  3$ 的张量，存储了所有视锥体网格点在 3D 空间中的坐标，即所有视锥体网格点被精确定位到 3D LiDAR/BEV 空间中。最后一个维度3就是存储的XYZ 3D坐标点

            # Load from the pkl
            """ import pickle
            with open("precomputed_features.pkl", "rb") as f:
                data = pickle.load(f) """
            #x  它已经被提升（Lift）成 $B N D H W C$ 的视锥体特征张量。
            x, est_depth_distr, gt_depth_distr, counts_3d = self.get_cam_feats(img, depth)  

            """ import pickle
            with open("depth_deploy.pkl", "rb") as f:
                data = pickle.load(f) """
            #根据 geom 中定义的 3D 坐标，将视锥体特征 x 中的每一个特征点**投影（或分散）**到预定义的 3D 网格（BEV 网格）的对应位置上。通常，如果多个视锥体特征点落在同一个 BEV 网格单元内，会使用求和或平均等方式进行聚合。
            # 最终结果 x 是一个 $B \times C \times H_{bev} \times W_{bev}$ 的特征图，即 鸟瞰图（BEV）特征。
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
            #这段是 估计深度分布 (est_depth_distr) 和 LiDAR 衍生的真值深度分布 (gt_depth_distr) 之间的交叉熵损失 (Cross-Entropy Loss)，核心思想是：只在有 LiDAR 点云数据（即有可靠深度真值）的地方计算损失，以指导模型学习正确的深度分布

            #计算有效掩码：counts_3d 表示每个视锥体单元格内 LiDAR 点的数量。对最后一个维度求和（$\text{sum}(\text{dim}=-1)$）后，如果点数大于 0，则标记为 True。mask_flat 是一个布尔向量，标记了所有包含至少一个 LiDAR 点的 3D 单元格
            mask_flat = counts_3d.sum(dim=-1).view(-1) > 0
            #展平真值分布：将 LiDAR 衍生的真值深度分布张量展平，形状变为 $(N_{\text{total}}, D)$，其中 $N_{\text{total}}$ 是所有视锥体单元格的总数，$D$ 是离散深度 bin 的数量
            gt_depth_distr_flat = gt_depth_distr.view(-1, self.D)
            est_depth_distr_flat = est_depth_distr.reshape(-1, self.D)  #将模型预测的深度分布展平，形状与真值分布一致。
            #计算交叉熵损失：对于每个视锥体单元格，计算真值分布和预测分布之间的交叉熵。然后使用 mask_flat 仅保留那些有 LiDAR 点的单元格的损失值，最后对这些损失求和并归一化，得到最终的深度损失
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

            #使用 scatter_add_ 高效地将 LiDAR 点的数量累加到对应的 3D 单元格 $\boldsymbol{(B, N, fH, fW, D)}$ 中。counts_3d 表示每个视锥体单元格内 LiDAR 点的密度
            counts_flat = torch.zeros(BN * fH * fW * self.D, dtype=torch.float, device=d.device)
            counts_flat.scatter_add_(
                0, flat_index, torch.ones_like(flat_index, dtype=torch.float, device=flat_index.device)
            )

            counts_3d = counts_flat.view(B, N, fH, fW, self.D)
            counts_3d[..., 0] = 0.0

            # mask_flat = counts_3d.sum(dim=-1).view(-1) > 0

            # gt_depth_distr = torch.softmax(counts_3d, dim=-1)
            gt_depth_distr = counts_3d / (counts_3d.sum(dim=-1, keepdim=True) + 1e-8) #对计数结果沿深度 $D$ 轴进行归一化（求 Softmax 替代），得到概率分布 $P(d|x, y)$。这即是用于深度损失监督的真值
            # gt_depth_distr_flat = gt_depth_distr.view(-1, self.D)
            # =================== TEST
        else:
            gt_depth_distr = None
            counts_3d = None
#于对输入的稀疏 LiDAR 深度图 $d$ 进行预处理和特征提取。输入 d: 原始 LiDAR 深度图，形状通常为 $(B \cdot N, 1, H, W)$，其中 $1$ 是通道数。多尺度降维: 该网络通过步长为 4 和 2 的卷积层，对深度图进行快速下采样，并增加通道数（$1 \rightarrow 64$）。输出 d_feat: 预处理后的深度特征图，其空间尺寸（$H, W$）会降采样到与图像特征图 $(fH, fW)$ 相同，通道数为 64
        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        #输出部分: 网络输出的前 $D$ 个通道（即 $output[:, :D]$）。作用: 这 ${D}$ 个通道代表每个 2D 像素在 $D$ 个离散深度 ${bin}$ 上的对数几率 (Logits)。在 get_cam_feats 中，这部分输出会经过 ${softmax}$ 归一化，得到 ${est\_depth\_distr}$，用于深度监督和特征提升的权重
        #输出部分: 网络的输出的后 ${C}$ 个通道（即 ${output}[:, D:(D+C)]$）。作用: 这 ${C}$ 个通道代表经过 depthnet 精炼和上下文融合后的图像特征。这些特征随后会与深度分布进行外积，实现 2D 特征到 3D 视锥体的提升 (Lift)
        x = self.depthnet(x)  #输出通道数是 ${D + C}$

        depth = x[:, : self.D].softmax(dim=1)
        est_depth_distr = depth.permute(0, 2, 3, 1).reshape(B, N, fH, fW, self.D) #估计深度分布: 网络输出的前 $D$ 个通道通过 ${softmax}$ 得到每个 2D 特征点在 $D$ 个深度 bin 上的估计概率分布 $Q}(d|x, y)$

        #这一步（通常称为 Guided Backpropagation 或 Auxiliary Loss Injection）用于辅助训练。它将真值分布 (${gt\_depth\_distr}$) 的信息（通过 ${torch.maximum}$)注入到估计分布的梯度流中。由于使用了 .detach()，这种注入只影响梯度的计算，而不影响前向传播的值，旨在提供更稳定的深度梯度
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
        #深度权重: depth.unsqueeze(1) 是估计的 $D$ 个深度概率，作为权重。特征: x[:, self.D : (self.D + self.C)] 是图像特征 $C$。外积: 两者相乘（通过广播机制实现外积）将 2D 图像特征 $C$ 沿着 深度 $D$ 轴 进行复制和加权。结果是一个 3D 视锥体特征 张量
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x, est_depth_distr, gt_depth_distr, counts_3d
    
    def forward(self, *args, **kwargs):
        x, depth_loss = super().forward(*args, **kwargs)
        x = self.downsample(x)
        return x, depth_loss
