#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : module for hebbian learning based orthogonal projection
import torch
import torch.nn as nn


class HLOP(nn.Module):
    def __init__(self, in_features, lr=0.01, momentum=True, spiking=False, spiking_scale=20., spiking_timesteps=1000.):
        super(HLOP, self).__init__()
        self.n_subspace = 0  # 子空间数量

        self.in_features = in_features  # 输入特征维度
        self.out_features_list = []  # 记录每个子空间的输出特征维度

        self.index_list = [0]  # 记录了每个子空间在权重矩阵中的起始和结束行索引

        self.weight = None  # HLOP权重矩阵，将输入特征 x 映射到一个新的输出空间（一个子空间）。
        self.momentum = momentum
        if self.momentum:
            self.delta_weight_momentum = None
            self.m = 0.9

        self.spiking = spiking  # 是否启用尖峰模式
        self.spiking_scale = spiking_scale  # 控制尖峰模式的尺度
        self.spiking_timesteps = spiking_timesteps  # 控制尖峰模式的时间步

        self.lr = lr  # 学习率

    def add_subspace(self, out_features: int):
        """
        增加子空间
        Args:
            out_features: 输出特征维度

        Returns:
            更新子空间的对应的权重矩阵的值
        """
        assert out_features > 0
        self.n_subspace += 1  # 子空间数量+1
        self.out_features_list.append(out_features)
        self.index_list.append(self.index_list[-1] + out_features)

        # 如果只有一个子空间 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.n_subspace == 1:
            self.weight = nn.Parameter(torch.zeros((out_features, self.in_features)))  # 全零权重矩阵
            if self.momentum:
                self.delta_weight_momentum = nn.Parameter(torch.zeros((out_features, self.in_features)))
            torch.nn.init.orthogonal_(self.weight.data)  # 正交初始化权重矩阵
        # 如果不止一个子空间 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        else:
            out_features_new = self.weight.size(0) + out_features  # 获取最新的输出特征
            weight_new = torch.zeros((out_features_new, self.in_features)).to(self.weight.device)
            weight_new[: self.weight.size(0), :] = self.weight.data  # 获取之前的权重矩阵的值
            torch.nn.init.orthogonal_(weight_new[self.weight.size(0):, :])  # 正交初始化权重矩阵

            if self.momentum:
                delta_weight_momentum_new = torch.zeros((out_features_new, self.in_features)).to(self.weight.device)
                delta_weight_momentum_new[: self.weight.size(0), :] = self.delta_weight_momentum.data  # 获取之前的值
                self.delta_weight_momentum = nn.Parameter(delta_weight_momentum_new)  # 更新

            self.weight = nn.Parameter(weight_new)  # 更新权重矩阵

    def merge_subspace(self):
        """
        合并所有子空间为一个整体，简化管理
        Returns:
            合并后的相关参数
        """
        assert self.n_subspace > 0
        self.n_subspace = 1  # 子空间数量为1
        self.out_features_list = [self.index_list[-1]]  # 合并了输出维度
        self.index_list = [0, self.out_features_list[0]]  # 合并了权重矩阵的起始位置

    def update_weights(self, x, y, xhat, fix_subspace_id_list=None):
        """
        根据Hebbian规则更新权重矩阵
        Args:
            x: 输入
            y: 输出
            xhat: 重构误差
            fix_subspace_id_list: 限制权重不更新的特定子空间的id列表

        Returns:

        """
        # x: [batch, N dim], y: [batch, M dim], weight: [M dim * N dim]
        weight = self.weight.data
        if self.momentum:
            delta_weight_momentum = self.delta_weight_momentum.data

        out_features, in_features = weight.size()
        assert in_features == x.size(1) and out_features == y.size(1)

        # 获取权重矩阵中固定不变的部分
        fix_index = []
        if fix_subspace_id_list is not None:
            for sid in fix_subspace_id_list:
                fix_index.extend(range(self.index_list[sid], self.index_list[sid + 1]))

        # 计算权重矩阵的变化量
        delta_weight = (torch.mm(y.t(), x - xhat) / x.shape[0])
        delta_weight = torch.clamp(delta_weight, -10, 10)
        delta_weight[fix_index, :] = 0.  # 固定不变部分的变化量设置为0

        if self.momentum:
            fix_term = delta_weight_momentum[fix_index, :]
            delta_weight_momentum[fix_index, :] = 0
            delta_weight_momentum = self.m * delta_weight_momentum + (1 - self.m) * delta_weight
            weight = weight + self.lr * delta_weight_momentum
            delta_weight_momentum[fix_index, :] = fix_term
        else:
            weight = weight + self.lr * delta_weight

        self.weight.data = weight
        if self.momentum:
            self.delta_weight_momentum.data = delta_weight_momentum

    def set_subspace(self, weight, id_list=[0]):
        """
        设置子空间的值
        Args:
            weight:
            id_list:

        Returns:

        """
        index = []
        for i in id_list:
            index.extend(range(self.index_list[i], self.index_list[i + 1]))
        self.weight.data[index, :] = weight.clone()

    def get_weight_value(self, id_list=[0]):
        """
        获取子空间的值
        Args:
            id_list:

        Returns:

        """
        index = []
        for i in id_list:
            index.extend(range(self.index_list[i], self.index_list[i + 1]))
        weight_ = self.weight.data[index, :].clone()
        return weight_

    def inference(self, x, subspace_id_list=[0]):
        """
        正向推理操作，将x通过权重矩阵后映射为y
        Args:
            x: 原始的数据
            subspace_id_list: 子空间id列表

        Returns:
            经过权重矩阵映射后的数据
        """
        # 根据给定的 subspace_id_list，构建一个包含目标子空间行索引的列表 index
        # 这些索引用来从权重矩阵中提取对应子空间的权重
        index = []
        for sid in subspace_id_list:
            index.extend(range(self.index_list[sid], self.index_list[sid + 1]))

        weight = self.weight.data[index, :]

        y0 = torch.mm(x, weight.t())
        y = y0

        if self.spiking:
            y = ((torch.clamp(y, -self.spiking_scale, self.spiking_scale) /
                  self.spiking_scale * self.spiking_timesteps).round() /
                 self.spiking_timesteps * self.spiking_scale)

        return y

    def inference_back(self, y, subspace_id_list=[0]):
        """
        正向推理操作，将x通过权重矩阵后映射为y
        Args:
            y: 经过权重矩阵映射后的数据
            subspace_id_list: 子空间id列表

        Returns:
            原始的数据
        """
        index = []
        for sid in subspace_id_list:
            index.extend(range(self.index_list[sid], self.index_list[sid + 1]))

        weight = self.weight.data[index, :]

        x = torch.mm(y, weight)

        return x

    def projection(self, x, subspace_id_list=[0]):
        """
        将输入x投影到指定的子空间
        Args:
            x: 输入
            subspace_id_list:指定的子空间id列表

        Returns:
            投影到指定的子空间后的x
        """
        y = self.inference(x, subspace_id_list)
        x_proj = self.inference_back(y, subspace_id_list)

        return x_proj

    def forward_with_update(self, x, iteration=5, fix_subspace_id_list=None):
        subspace_id_list = list(range(self.n_subspace))
        # 多次迭代进行权重矩阵更新
        for i in range(iteration):
            y = self.inference(x, subspace_id_list)
            xhat = self.inference_back(y, subspace_id_list)
            self.update_weights(x, y, xhat, fix_subspace_id_list)  # 更新权重矩阵更新

    def projection_with_update(self, x, iteration=5, subspace_id_list=[0], fix_subspace_id_list=None):
        x_proj = self.projection(x, subspace_id_list)
        self.forward_with_update(x, iteration, fix_subspace_id_list)

        return x_proj

    def get_proj_func(self, iteration=5, subspace_id_list=[0], forward_with_update=False, fix_subspace_id_list=None):
        if forward_with_update:
            return lambda x: self.projection_with_update(x, iteration, subspace_id_list, fix_subspace_id_list)
        else:
            return lambda x: self.projection(x, subspace_id_list)

    def adjust_lr(self, gamma):
        self.lr = self.lr * gamma
