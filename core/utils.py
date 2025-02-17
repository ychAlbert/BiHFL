#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from torch import nn
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------------------------------------------------
# 客户端数据集处理相关函数
# ----------------------------------------------------------------------------------------------------------------------
class GeneralDataset(Dataset):
    def __init__(self, data, labels, n_class):
        self.data = data
        self.labels = labels
        self.n_class = n_class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label


def split_trainset_by_dirichlet(args, xtrain, ytrain, taskcla):
    print('\n\033[93m正在分配训练集\033[0m')
    # 获取客户端个数
    n_client = args.n_client

    client_datasets = []
    for task in taskcla:
        task_id, task_n_class = task[0], task[1]

        # partition[i][j] 是客户端j从类别i中获得的样本的百分比
        partition = np.random.dirichlet(args.dirichlet_concentration, size=task_n_class)

        labels_sorted = ytrain[task_id].sort()
        class_index_list = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))

        # 存放每个类别的标签索引的字典
        class_index_map = defaultdict(list)
        for cla, idx in class_index_list:
            class_index_map[cla].append(idx)

        dict_clients = defaultdict(list)
        # 划分数据给客户端
        # 统计每个客户端的数据分布，即每个客户端有多少张各种类别的图片
        class_ndata_map_for_clients = []

        # 对于每一个类别
        for i in range(task_n_class):
            n_data_of_class = len(class_index_map[i])
            # 向所有的客户端划分图片
            ndata_for_clients = []
            # 对于每一个客户端
            for client_id in range(n_client):
                # 第j个客户端从某类数据中分得的数量
                n_data_of_client = int(partition[i][client_id] * n_data_of_class)
                if n_data_of_client > 0:
                    ndata_for_clients.append(n_data_of_client)
                else:
                    ndata_for_clients.append(0)
                dict_clients[client_id] += class_index_map[i][: n_data_of_client]
                del class_index_map[i][: n_data_of_client]
            class_ndata_map_for_clients.append(ndata_for_clients)
            # 给客户端0进行补全
            dict_clients[0] += class_index_map[i]

        class_ndata_map_for_clients = np.array(class_ndata_map_for_clients)
        class_ndata_map_for_clients = class_ndata_map_for_clients.T

        n_data_of_client = np.sum(class_ndata_map_for_clients, axis=1).tolist()
        n_data_of_classes = np.sum(class_ndata_map_for_clients, axis=0).tolist()
        n_data_of_client_classes = class_ndata_map_for_clients.tolist()

        print('任务：{}'.format(task_id))

        print(f'     类别情况：')
        print('          ', end='')
        for i in range(len(n_data_of_classes)):
            end_char = ', ' if i < len(n_data_of_classes) - 1 else ''
            print(f'类别{i}({n_data_of_classes[i]})', end=end_char)

        print(f'\n     本地情况：')
        for i in range(len(n_data_of_client)):
            print(f'          本地{i}(总数据量 {n_data_of_client[i]}) 数据分布：{n_data_of_client_classes[i]}')

        client_task_datasets = []
        for k in range(n_client):
            client_data_idx = dict_clients[k]
            client_data_x = xtrain[task_id][client_data_idx]
            client_data_y = ytrain[task_id][client_data_idx]
            client_dataset = GeneralDataset(data=client_data_x, labels=client_data_y, n_class=task_n_class)
            client_task_datasets.append(client_dataset)
        client_datasets.append(client_task_datasets)

    splitted_trainsets = []
    for i in range(n_client):
        splitted_trainset = {}
        for task in taskcla:
            splitted_trainset[f'task {task[0]}'] = client_datasets[task[0]][i]
        splitted_trainsets.append(splitted_trainset)
    print('\033[93m训练集分配完毕\033[0m\n')
    return splitted_trainsets


def calculate_distribution_by_emd(emd_target, num_classes):
    def objective(x):
        # 计算列表元素与均匀分布之间的平方差和再开根
        norm_diff = abs(np.sqrt(np.sum((x - [0.1] * len(x)) ** 2)) - emd_target)
        return norm_diff

    def constraint1(x):
        # 约束条件：列表元素之和为1
        return np.sum(x) - 1.0

    def constraint2(x):
        # 约束条件：列表元素在[0, 1]内
        return np.array(x)

    def adjust_sum(arr):
        """
        微调数据，以防四舍五入时会丢失一点精度。
        """
        diff = 1 - np.sum(arr)  # 计算与目标和的差值
        random_index = np.random.randint(0, len(arr))
        arr[random_index] += diff  # 将差值加到随机素上
        return arr

    # 随机生成初始猜测值
    x0 = np.random.rand(num_classes)

    # 设置约束条件
    cons = [{'type': 'eq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2}]

    # 进行优化
    sol = minimize(objective, x0, method='SLSQP', constraints=cons)

    if sol.success:
        return True, adjust_sum(sol.x)
    else:
        return False, None


def split_trainset_by_emd(args, xtrain, ytrain, taskcla):
    print('\n\033[93m正在分配训练集\033[0m')
    # 获取客户端个数
    n_client = args.n_client

    client_datasets = []
    for task in taskcla:
        task_id, task_n_class = task[0], task[1]

        partition = np.zeros((task_n_class, n_client), dtype=np.float16)
        for client_id in range(n_client):
            done, distribution = False, []
            while not done:
                done, distribution = calculate_distribution_by_emd(args.emd_distance[client_id], task_n_class)
            partition[:, client_id] = np.array(distribution)

        for class_id in range(task_n_class):
            partition[class_id][:] = partition[class_id][:] / np.sum(partition[class_id][:])
            partition[class_id][-1] = 1.0 - np.sum(partition[class_id][:-1])

        labels_sorted = ytrain[task_id].sort()
        class_index_list = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))

        # 存放每个类别的标签索引的字典
        class_index_map = defaultdict(list)
        for cla, idx in class_index_list:
            class_index_map[cla].append(idx)

        dict_clients = defaultdict(list)
        # 划分数据给客户端
        # 统计每个客户端的数据分布，即每个客户端有多少张各种类别的图片
        class_ndata_map_for_clients = []

        # 对于每一个类别
        for i in range(task_n_class):
            n_data_of_class = len(class_index_map[i])
            # 向所有的客户端划分图片
            ndata_for_clients = []
            # 对于每一个客户端
            for client_id in range(n_client):
                # 第j个客户端从某类数据中分得的数量
                n_data_of_client = int(partition[i][client_id] * n_data_of_class)
                if n_data_of_client > 0:
                    ndata_for_clients.append(n_data_of_client)
                else:
                    ndata_for_clients.append(0)
                dict_clients[client_id] += class_index_map[i][: n_data_of_client]
                del class_index_map[i][: n_data_of_client]
            class_ndata_map_for_clients.append(ndata_for_clients)
            # 给客户端0进行补全
            dict_clients[0] += class_index_map[i]

        class_ndata_map_for_clients = np.array(class_ndata_map_for_clients)
        class_ndata_map_for_clients = class_ndata_map_for_clients.T

        n_data_of_client = np.sum(class_ndata_map_for_clients, axis=1).tolist()
        n_data_of_classes = np.sum(class_ndata_map_for_clients, axis=0).tolist()
        n_data_of_client_classes = class_ndata_map_for_clients.tolist()

        print('任务：{}'.format(task_id))

        print(f'     类别情况：')
        print('          ', end='')
        for i in range(len(n_data_of_classes)):
            end_char = ', ' if i < len(n_data_of_classes) - 1 else ''
            print(f'类别{i}({n_data_of_classes[i]})', end=end_char)

        print(f'\n     本地情况：')
        for i in range(len(n_data_of_client)):
            print(f'          本地{i}(总数据量 {n_data_of_client[i]}) 数据分布：{n_data_of_client_classes[i]}')

        client_task_datasets = []
        for k in range(n_client):
            client_data_idx = dict_clients[k]
            client_data_x = xtrain[task_id][client_data_idx]
            client_data_y = ytrain[task_id][client_data_idx]
            client_dataset = GeneralDataset(data=client_data_x, labels=client_data_y, n_class=task_n_class)
            client_task_datasets.append(client_dataset)
        client_datasets.append(client_task_datasets)

    splitted_trainsets = []
    for i in range(n_client):
        splitted_trainset = {}
        for task in taskcla:
            splitted_trainset[f'task {task[0]}'] = client_datasets[task[0]][i]
        splitted_trainsets.append(splitted_trainset)
    print('\033[93m训练集分配完毕\033[0m\n')
    return splitted_trainsets


def split_trainset_by_iid(args, xtrain, ytrain, taskcla):
    print('\n\033[93m正在分配训练集\033[0m')
    n_client = args.n_client
    client_datasets = []

    for task in taskcla:
        task_id, task_n_class = task[0], task[1]
        data_per_client = len(xtrain[task_id]) // n_client

        client_task_datasets = []
        for i in range(n_client):
            start_idx = i * data_per_client
            end_idx = (i + 1) * data_per_client if i != n_client - 1 else len(xtrain[task_id])
            client_data_x = xtrain[task_id][start_idx:end_idx]
            client_data_y = ytrain[task_id][start_idx:end_idx]
            client_dataset = GeneralDataset(data=client_data_x, labels=client_data_y, n_class=task_n_class)
            client_task_datasets.append(client_dataset)
        client_datasets.append(client_task_datasets)

    splitted_trainsets = []
    for i in range(n_client):
        splitted_trainset = {}
        for task in taskcla:
            splitted_trainset[f'task {task[0]}'] = client_datasets[task[0]][i]
        splitted_trainsets.append(splitted_trainset)
    print('\033[93m训练集分配完毕\033[0m\n')
    return splitted_trainsets


# ----------------------------------------------------------------------------------------------------------------------
# 指标评估相关类和函数
# ----------------------------------------------------------------------------------------------------------------------
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# ----------------------------------------------------------------------------------------------------------------------
# 模型处理相关函数
# ----------------------------------------------------------------------------------------------------------------------
def reset_net(net: nn.Module):
    """
    将网络的状态重置。做法是遍历网络中的所有 ``Module``，若含有 ``reset()`` 函数，则调用。
    @param net: 任何属于 ``nn.Module`` 子类的网络
    @return:
    """
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()


