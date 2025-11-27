# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from .voxel_layer import dynamic_voxelize, hard_voxelize


class _Voxelization(Function):

    @staticmethod
    def forward(ctx, points, voxel_size, coors_range, max_points=35, max_voxels=20000, deterministic=True):
        """convert kitti points(N, >=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        if max_points == -1 or max_voxels == -1: #通常用于 PointPillar 或其他不需要限制每个体素内点数的模型。动态体素化 (Dynamic Voxelization)
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int) #初始化坐标：创建一个与输入点数量相同，维度为 3 的坐标张量
            dynamic_voxelize(points, coors, voxel_size, coors_range, 3)  #执行体素化：调用底层的 dynamic_voxelize 函数（C++/CUDA 实现）。这个函数直接计算 每个输入点 属于哪个体素，并将该体素的离散坐标 $(X, Y, Z)$ 写入 coors
            return coors
        else:
            voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1))) #初始化体素张量：预分配一个固定大小的张量来存储体素特征。尺寸为 $\text{(最大体素数, 单个体素最大点数, 特征维度)}$
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)
#执行体素化：调用底层的 hard_voxelize 函数（C++/CUDA 实现）。它完成以下任务：1. 计算每个点所属的体素坐标。2. 将点云数据复制到 voxels 预分配的张量中。
#3. 硬限制：每个体素只保留最多 max_points 个点。如果体素已满或达到 max_voxels 限制，点会被丢弃（因此用户通常需要打乱点云）
            voxel_num = hard_voxelize(
                points,
                voxels,
                coors,
                num_points_per_voxel,
                voxel_size,
                coors_range,
                max_points,
                max_voxels,
                3,
                deterministic,
            )
            # select the valid voxels
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
            #输出：返回经过裁切的三个张量：1. voxels: 包含点的体素特征 ($M \times \text{max\_points} \times C$)。2. coors: 稀疏体素坐标 ($M \times 3$ 的 $(X, Y, Z)$ 索引)。
            # 3. num_points_per_voxel: 每个有效体素内的点数 ($M$)
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class Voxelization(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000, deterministic=True):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.voxel_size = voxel_size # [x, y, z]定义每个体素在 3D 空间中的边长（分辨率）
        self.point_cloud_range = point_cloud_range #定义点云数据在 3D 空间中的范围，格式为 [x_min, y_min, z_min, x_max, y_max, z_max] 超出此范围的点会被丢弃
        self.max_num_points = max_num_points #每个体素中允许包含的最大点数
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels  #  限制在训练和测试阶段生成的最大体素数量。这是为了内存管理和固定计算图。
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic  #控制是否使用确定性（通常较慢）或非确定性（通常较快）的体素化实现

        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32) 
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size #计算在给定点云范围和体素大小下，整个 3D 空间可以划分成多少个体素单元
        grid_size = torch.round(grid_size).long() 
        input_feat_shape = grid_size[:2] 
        self.grid_size = grid_size 
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w] removed
        self.pcd_shape = [*input_feat_shape, 1]  # [::-1]

    def forward(self, input):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]
#实际执行体素化.调用底层的 voxelization 函数后，它完成了两个关键任务：点到体素的映射： 根据 voxel_size 和 point_cloud_range，计算出每个点 $(x, y, z)$ 所在的离散体素索引 $(i_x, i_y, i_z)$。
#特征聚合： 将所有落入同一个体素的点的特征进行聚合（如求和或平均）。
#底层 voxelization 函数的输出，通常是：体素特征 (feats): 聚合后的点特征。体素坐标 (coords): 非空体素的离散索引 $(i_x, i_y, i_z)$。点数 (sizes): 每个体素内的点数量
        return voxelization(
            input,
            self.voxel_size,
            self.point_cloud_range,
            self.max_num_points,
            max_voxels,
            self.deterministic,
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "voxel_size=" + str(self.voxel_size)
        tmpstr += ", point_cloud_range=" + str(self.point_cloud_range)
        tmpstr += ", max_num_points=" + str(self.max_num_points)
        tmpstr += ", max_voxels=" + str(self.max_voxels)
        tmpstr += ", deterministic=" + str(self.deterministic)
        tmpstr += ")"
        return tmpstr
