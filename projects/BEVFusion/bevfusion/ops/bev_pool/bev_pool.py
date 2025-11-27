import torch
from torch.onnx.symbolic_helper import _get_tensor_dim_size, _get_tensor_sizes

from . import bev_pool_ext


class QuickCumsum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None
    







class QuickCumsumTrainingCuda(torch.autograd.Function):
    # 图像特征点乘深度分数的体素特征([1695595, 80]),排序和过滤后的真实模版三维坐标点的体素索引和batch——id([1695595, 4]), 每个点体素序号([1695595),1, 1,360,360
    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        #由于特征 (x) 已经根据 ranks 排序，所有属于同一个 BEV 单元的特征是连续排列的。
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool) #kept 标记了每个新的 BEV 单元特征块的起始位置。torch.ones(...) 生成一个 全 True 的布尔张量
        kept[1:] = ranks[1:] != ranks[:-1]   # kept[i]=Trueiff ranks[i]=ranks[i−1](i>0) 标记每个 voxel块起始点的位置 ([ True,  True, False,  ..., False, False,  True]
        interval_starts = torch.where(kept)[0].int()  #就是这些起始位置的索引 interval_starts([ 0,1, 5,  ..., 1379475, 1379483, 1379493]
        interval_lengths = torch.zeros_like(interval_starts) 
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]  #每个 voxel 的点数 = 下一个 voxel 起始点 - 当前 voxel 起始点 interval_lengths[i]=interval_starts[i+1]−interval_starts[i]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]  #最后一个单元的长度是总特征数减去其起始点
        geom_feats = geom_feats.int()
        #这是实际的聚合发生的地方 bev_pool_forward是一个在 GPU 上高度优化的 C++/CUDA 函数
        #它利用 interval_starts 和 interval_lengths 的信息，并行地对每个 BEV 单元的特征向量进行求和 (或平均)
        out = bev_pool_ext.bev_pool_forward(
            x, # 图像特征点乘深度分数的体素特征([1695595, 80])
            geom_feats,# 排序和过滤后的真实模版三维坐标点的体素索引和batch—id([1695595, 4])
            interval_lengths,# 每个 voxel 在 x 中连续点的起始位置和长度
            interval_starts,
            B,# 1
            D,# 1
            H,# 360
            W,# 360
        )
        #为了进行反向传播，必须保存 BEV 单元的边界信息（起始和长度），因为它们决定了聚合的映射关系
        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)# 前向传播完成后，反向传播需要知道每个 voxel 内哪些点被聚合
        # GPU kernel 在反向传播时，会把 voxel 对应的梯度再分配回原来的点
        ctx.saved_shapes = B, D, H, W
        return out # ([1, 1, 360, 360, 80]) B D H W C

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors #从前向传播中恢复 BEV 单元的边界信息
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )    #输入: 聚合后 BEV 特征的梯度 (out_grad)，bev_pool_backward 函数执行梯度的分散 (De-Splatting)。它将聚合后的 BEV 单元的梯度 (out_grad) 复制并分配回所有参与该单元聚合的原始视锥体特征 (x) 对应的位置。例如，如果 5 个视锥体特征向量聚合到了 BEV 单元 $A$，那么 BEV 单元 $A$ 上的梯度会复制 5 份，分别作为这 5 个原始特征向量的梯度。

        return x_grad, None, None, None, None, None, None


class QuickCumsumCuda(torch.autograd.Function):

    @staticmethod
    def symbolic(
        g,
        x,
        geom_feats,
        interval_lengths,
        interval_starts,
        B,
        D,
        H,
        W,
    ):
        output = g.op(
            "autoware::QuickCumsumCuda",
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            batch_size_i=B,
            dimension_i=D,
            height_i=H,
            width_i=W,
            outputs=1,
        )

        features_shape = _get_tensor_sizes(x)
        if features_shape is not None and hasattr(x.type(), "with_sizes"):
            output_type = x.type().with_sizes([B, D, H, W, _get_tensor_dim_size(x, -1)])
            output.setType(output_type)

        return output

    @staticmethod
    def forward(ctx, x, geom_feats, interval_lengths, interval_starts, B, D, H, W):
        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )
        return out

    @staticmethod
    def backward(ctx, out_grad):
        raise NotImplementedError


def bev_pool(feats, coords, ranks, B, D, H, W, is_training):
    assert feats.shape[0] == coords.shape[0]

    # NOTE(knzo25): we want to put all the operations we can in the graph
    #将视锥体特征 (feats) 根据其在 3D 空间中的位置 (coords，已排序的 ranks)，聚合到 BEV 网格中
    if is_training:
        x = QuickCumsumTrainingCuda.apply(feats, coords, ranks, B, D, H, W)
        # 返回一个聚合后的张量（ B x D x C x H x W）这个结果会参与计算图以便反向传播
    else:

        kept = torch.ones(feats.shape[0], device=feats.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = feats.shape[0] - interval_starts[-1]

        if coords.dtype != torch.int32:
            coords = coords.int()

        x = QuickCumsumCuda.apply(
            feats, coords, interval_lengths, interval_starts, int(B), D.item(), H.item(), W.item()
        )
    # ([1, 1, 360, 360, 80]) B D H W C -->([1, 80, 1, 360, 360])
    x = x.permute(0, 4, 1, 2, 3).contiguous()    #B C D H W 

    return x
