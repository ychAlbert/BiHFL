#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
from collections import defaultdict
from typing import List

import numpy as np
from scipy.optimize import minimize
import torch
from torch import nn
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------------------------------------------------
# 客户端数据集处理相关函数
# ----------------------------------------------------------------------------------------------------------------------
class GeneralDataset(Dataset):
    def __init__(self, data, label, num_classes):
        self.data = data
        self.labels = label
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label


def calculate_distribution_by_emd(emd_target, num_data, num_classes):
    """
    根据emd值计算数据集分布
    @param emd_target:
    @param num_data:
    @param num_classes:
    @return:
    """

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
        arr = [int(np.floor(i)) for i in arr]
        diff = num_data - np.sum(arr)  # 计算与目标和的差值
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
        return True, adjust_sum(np.round(sol.x * num_data))
    else:
        return False, None


def calculate_emd_by_distribution(distribution):
    num_classes = len(distribution)
    total_data_num = sum(distribution)
    iid_distribution = [1 / num_classes] * num_classes

    sum_squares = 0
    for i in range(num_classes):
        sum_squares += (distribution[i]/ total_data_num - iid_distribution[i]) ** 2
    return np.sqrt(sum_squares)


def calculate_indexes_by_distribution(distribution, classes_indexes, num_data_used):
    indexes = []
    num_classes = len(num_data_used)

    for i in range(num_classes):
        if distribution[i] > 0:
            left = int(num_data_used[i])
            right = int(num_data_used[i] + distribution[i])
            indexes.extend(classes_indexes[i][left:right])
            num_data_used[i] += distribution[i]

    return indexes

def distribute_data_dirichlet(dataset, args, n_class=10):
    np.random.seed(args.seed)
    num_clean_agents = args.num_users
    print(args.concent)
    # partition[c][i] is the fraction of samples agent i gets from class
    partition = np.random.dirichlet([args.concent] * num_clean_agents, size=n_class)
    # print(partition)

    labels_sorted = dataset.targets.sort()
    class_by_labels = list(zip(labels_sorted.values.tolist(), labels_sorted.indices.tolist()))
    # convert list to a dictionary, e.g., at labels_dict[0], we have indexes for class 0
    # labels_dict[0]：所有0类的数据，此时只是将所有同类的数据放在一起了，还没有将他们划分给客户端
    labels_dict = defaultdict(list)
    for k, v in class_by_labels:
        labels_dict[k].append(v)
    # print(labels_dict.keys())#数据集标签

    dict_users = defaultdict(list)
    # 划分数据给客户端
    # 统计每个客户端的数据分布，即每个客户端有多少张各种类别的图片
    pic_distribution_every_client = []
    for c in range(n_class):
        # num of samples of class c in dataset 某类图片的总量
        n_classC_items = len(labels_dict[c])
        # 向所有的客户端划分图片
        pic_distribution_one_client = []
        for i in range(num_clean_agents):
            # num. of samples agent i gets from class c 第i个客户端从某类图片中分得的数量
            n_agentI_items = int(partition[c][i] * n_classC_items)
            if n_agentI_items > 0:
                pic_distribution_one_client.append(n_agentI_items)
            else:
                pic_distribution_one_client.append(0)
            dict_users[i] += labels_dict[c][:n_agentI_items]
            del labels_dict[c][:n_agentI_items]
        pic_distribution_every_client.append(pic_distribution_one_client)
        # if any class c item remains due to flooring, give em to first agent 分剩的都给第0个客户端
        dict_users[0] += labels_dict[c]
        pic_distribution_one_client[0] += len(labels_dict[c])

    pic_distribution_every_client = np.array(pic_distribution_every_client)
    pic_distribution_every_client = pic_distribution_every_client.T
    # print(dict_users)
    # print("每个客户端的数据分布：")
    # print(pic_distribution_every_client)
    sum_res = np.sum(pic_distribution_every_client, axis=1)
    sum_ = np.sum(pic_distribution_every_client, axis=0)
    # print("每个客户端的总数据量：{}".format(sum_res))

    return dict_users, pic_distribution_every_client, sum_res, sum_

def split_trainset_by_emd(xtrain, ytrain, taskcla: List[tuple], emd_targets: List[list], num_data: List[list]):
    each_class_indexes = [None] * len(taskcla)
    num_data_used = [None] * len(taskcla)

    for task in taskcla:
        task_id = task[0]
        task_num_classes = task[1]
        # 获取任务的数据集
        each_class_indexes[task_id] = {key: torch.where(torch.tensor(trainset[f'task {task_id}'].labels == key))[0].cpu().detach().numpy() for key in range(task_num_classes)}
        num_data_used[task_id] = [0]*task_num_classes

    num_clients = len(emd_targets)
    splitted_trainsets = []
    emd_reals = []
    for i in range(num_clients):
        print(f'trainset of client {i}: ')
        # 客户端i的训练集
        splitted_trainset = {}
        # 客户端i的实际emd
        client_emd_real = []

        for task in taskcla:
            task_id = task[0]
            task_num_classes = task[1]
            print(f'    task {task_id} distribution: ', end='')
            # 计算客户端i的task任务的数据集分布
            done, distribution = False, []
            while not done:
                done, distribution = calculate_distribution_by_emd(emd_targets[i][task_id], num_data[i][task_id], task_num_classes)
            print(distribution)

            indexes = calculate_indexes_by_distribution(distribution, each_class_indexes[task_id], num_data_used[task_id])
            splitted_trainset[f'task {task_id}'] = torch.utils.data.Subset(trainset[f'task {task_id}'], indexes)
            emd_real = calculate_emd_by_distribution(distribution)
            client_emd_real.append(emd_real)

        emd_reals.append(client_emd_real)
        splitted_trainsets.append(splitted_trainset)

    return splitted_trainsets, emd_reals


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


def model_parameter_vector(model):
    """
    将给定模型model的所有参数整合成一个一维的向量
    @param model: 给定模型
    @return: 整合后的一维向量
    """
    param = [param.view(-1) for param in model.parameters()]
    return torch.cat(param, dim=0)


# ----------------------------------------------------------------------------------------------------------------------
# 实验准备相关函数
# ----------------------------------------------------------------------------------------------------------------------
def prepare_bptt_ottt(experiment_name: str):
    """
    根据实验名称准备bptt和ottt的值
    @param experiment_name: 实验名称
    @return:
    """
    bptt, ottt = False, False
    if experiment_name.endswith('bptt'):
        bptt, ottt = True, False
    elif experiment_name.endswith('ottt'):
        bptt, ottt = False, True
    return bptt, ottt


def prepare_hlop_out(experiment_name: str):
    """
    根据实验名称准备hlop_out_XXX相关的值
    @param experiment_name:
    @return:
    """
    hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1 = [], [], []
    if experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
        hlop_out_num = [80, 200, 100]
        hlop_out_num_inc = [70, 70, 70]
    elif experiment_name == 'cifar':  # cifar 实验
        hlop_out_num = [6, 100, 200]
        hlop_out_num_inc = [2, 20, 40]
    elif experiment_name == 'miniimagenet':  # miniimagenet 实验
        hlop_out_num = [24, [90, 90], [90, 90], [90, 180, 10], [180, 180], [180, 360, 20], [360, 360], [360, 720, 40],
                        [720, 720]]
        hlop_out_num_inc = [2, [6, 6], [6, 6], [6, 12, 1], [12, 12], [12, 24, 2], [24, 24], [24, 48, 4], [48, 48]]
        hlop_out_num_inc1 = [0, [2, 2], [2, 2], [2, 4, 0], [4, 4], [4, 8, 0], [8, 8], [8, 16, 0], [16, 16]]
    elif experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
        hlop_out_num = [6, [40, 40], [40, 40], [40, 100, 6], [100, 100], [100, 200, 8], [200, 200], [200, 200, 16],
                        [200, 200]]
        hlop_out_num_inc = [6, [40, 40], [40, 40], [40, 100, 6], [100, 100], [100, 200, 8], [200, 200], [200, 200, 16],
                            [200, 200]]
    return hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1


def prepare_dataset(experiment_name: str, dataset_path: str, seed: int):
    """
    根据实验名称准备hlop_out_XXX相关的值
    @param experiment_name:
    @param dataset_path:
    @param seed:
    @return:
    """
    xtrain, ytrain, xtest, ytest, taskcla = {}, {}, {}, {}, None
    if experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
        from FLcore.dataloader import pmnist as pmd
        data, taskcla, inputsize = pmd.get(data_dir=dataset_path, seed=seed)
        for task_id, ncla in taskcla:
            xtrain[task_id] = data[task_id]['train']['x']
            ytrain[task_id] = data[task_id]['train']['y']
            xtest[task_id] = data[task_id]['test']['x']
            ytest[task_id] = data[task_id]['test']['y']
    elif experiment_name == 'cifar':  # cifar 实验
        from FLcore.dataloader import cifar100 as cf100
        data, taskcla, inputsize = cf100.get(data_dir=dataset_path, seed=seed)
        for task_id, ncla in taskcla:
            xtrain[task_id] = data[task_id]['train']['x']
            ytrain[task_id] = data[task_id]['train']['y']
            xtest[task_id] = data[task_id]['test']['x']
            ytest[task_id] = data[task_id]['test']['y']
    elif experiment_name == 'miniimagenet':  # miniimagenet 实验
        from FLcore.dataloader import miniimagenet as data_loader
        dataloader = data_loader.DatasetGen(data_dir=dataset_path, seed=seed)
        taskcla, inputsize = dataloader.taskcla, dataloader.inputsize
        for task_id, ncla in taskcla:
            data = dataloader.get(task_id)
            xtrain[task_id] = data[task_id]['train']['x']
            ytrain[task_id] = data[task_id]['train']['y']
            xtest[task_id] = data[task_id]['test']['x']
            ytest[task_id] = data[task_id]['test']['y']
    elif experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
        from FLcore.dataloader import five_datasets as data_loader
        data, taskcla, inputsize = data_loader.get(data_dir=dataset_path, seed=seed)
        for task_id, ncla in taskcla:
            xtrain[task_id] = data[task_id]['train']['x']
            ytrain[task_id] = data[task_id]['train']['y']
            xtest[task_id] = data[task_id]['test']['x']
            ytest[task_id] = data[task_id]['test']['y']
    return xtrain, ytrain, xtest, ytest, taskcla


def prepare_model(experiment_name: str, args, ncla):
    model = None
    if experiment_name == 'pmnist':  # pmnist 实验
        from FLcore.models import spiking_MLP
        snn_setting = {}
        snn_setting['timesteps'] = args.timesteps
        snn_setting['train_Vth'] = True if args.train_Vth == 1 else False
        snn_setting['Vth'] = args.Vth
        snn_setting['tau'] = args.tau
        snn_setting['delta_t'] = args.delta_t
        snn_setting['alpha'] = args.alpha
        snn_setting['Vth_bound'] = args.Vth_bound
        snn_setting['rate_stat'] = True if args.rate_stat == 1 else False
        hlop_with_wfr = True
        if args.not_hlop_with_wfr:
            hlop_with_wfr = False
        model = spiking_MLP(snn_setting, num_classes=ncla, n_hidden=800,
                            ss=args.sign_symmetric,
                            fa=args.feedback_alignment, hlop_with_wfr=hlop_with_wfr,
                            hlop_spiking=args.hlop_spiking,
                            hlop_spiking_scale=args.hlop_spiking_scale,
                            hlop_spiking_timesteps=args.hlop_spiking_timesteps)
    elif experiment_name == 'pmnist_bptt':  # pmnist_bptt 实验
        from FLcore.models import spiking_MLP_bptt
        model = spiking_MLP_bptt(num_classes=ncla, n_hidden=800, ss=args.sign_symmetric,
                                 fa=args.feedback_alignment, timesteps=args.timesteps,
                                 hlop_spiking=args.hlop_spiking, hlop_spiking_scale=args.hlop_spiking_scale,
                                 hlop_spiking_timesteps=args.hlop_spiking_timesteps)
    elif experiment_name == 'pmnist_ottt':  # pmnist_ottt 实验
        from FLcore.models import spiking_MLP_ottt
        model = spiking_MLP_ottt(num_classes=ncla, n_hidden=800, ss=args.sign_symmetric, fa=args.feedback_alignment,
                                 timesteps=args.timesteps, hlop_spiking=args.hlop_spiking,
                                 hlop_spiking_scale=args.hlop_spiking_scale,
                                 hlop_spiking_timesteps=args.hlop_spiking_timesteps)
    elif experiment_name == 'cifar':  # cifar 实验
        from FLcore.models import spiking_cnn
        snn_setting = {}
        snn_setting['timesteps'] = args.timesteps
        snn_setting['train_Vth'] = True if args.train_Vth == 1 else False
        snn_setting['Vth'] = args.Vth
        snn_setting['tau'] = args.tau
        snn_setting['delta_t'] = args.delta_t
        snn_setting['alpha'] = args.alpha
        snn_setting['Vth_bound'] = args.Vth_bound
        snn_setting['rate_stat'] = True if args.rate_stat == 1 else False
        hlop_with_wfr = True
        if args.not_hlop_with_wfr:
            hlop_with_wfr = False
        model = spiking_cnn(snn_setting, num_classes=ncla, ss=args.sign_symmetric,
                            hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking,
                            hlop_spiking_scale=args.hlop_spiking_scale,
                            hlop_spiking_timesteps=args.hlop_spiking_timesteps,
                            proj_type=args.hlop_proj_type)
    elif experiment_name == 'miniimagenet':  # miniimagenet 实验
        from FLcore.models import spiking_resnet18
        snn_setting = {}
        snn_setting['timesteps'] = args.timesteps
        snn_setting['train_Vth'] = True if args.train_Vth == 1 else False
        snn_setting['Vth'] = args.Vth
        snn_setting['tau'] = args.tau
        snn_setting['delta_t'] = args.delta_t
        snn_setting['alpha'] = args.alpha
        snn_setting['Vth_bound'] = args.Vth_bound
        snn_setting['rate_stat'] = True if args.rate_stat == 1 else False
        hlop_with_wfr = True
        if args.not_hlop_with_wfr:
            hlop_with_wfr = False
        model = spiking_resnet18(snn_setting, num_classes=ncla, nf=20, ss=args.sign_symmetric,
                                 hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking,
                                 hlop_spiking_scale=args.hlop_spiking_scale,
                                 hlop_spiking_timesteps=args.hlop_spiking_timesteps,
                                 proj_type=args.hlop_proj_type, first_conv_stride2=True)
    elif experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
        from FLcore.models import spiking_resnet18
        snn_setting = {}
        snn_setting['timesteps'] = args.timesteps
        snn_setting['train_Vth'] = True if args.train_Vth == 1 else False
        snn_setting['Vth'] = args.Vth
        snn_setting['tau'] = args.tau
        snn_setting['delta_t'] = args.delta_t
        snn_setting['alpha'] = args.alpha
        snn_setting['Vth_bound'] = args.Vth_bound
        snn_setting['rate_stat'] = True if args.rate_stat == 1 else False
        hlop_with_wfr = True
        if args.not_hlop_with_wfr:
            hlop_with_wfr = False
        if experiment_name == 'fivedataset':
            model = spiking_resnet18(snn_setting, num_classes=ncla, nf=20, ss=args.sign_symmetric,
                                     hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking,
                                     hlop_spiking_scale=args.hlop_spiking_scale,
                                     hlop_spiking_timesteps=args.hlop_spiking_timesteps,
                                     proj_type=args.hlop_proj_type)
        elif experiment_name == 'fivedataset_domain':
            model = spiking_resnet18(snn_setting, num_classes=ncla, nf=20, ss=args.sign_symmetric,
                                     hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking,
                                     hlop_spiking_scale=args.hlop_spiking_scale,
                                     hlop_spiking_timesteps=args.hlop_spiking_timesteps,
                                     proj_type=args.hlop_proj_type, share_classifier=True)
    return model
