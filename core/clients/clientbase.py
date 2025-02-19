#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : 联邦学习客户端的基础类
import copy
import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from core.utils import GeneralDataset

__all__ = ['Client']


class Client(object):
    def __init__(self, args, id, trainset, taskcla, model):
        self.args = args
        self.id = id  # id标识
        self.device = args.device
        self.taskcla = taskcla
        self.local_tasks = [eval(task.replace("task ", "")) for task in trainset.keys()]

        self.trainset = trainset
        self.train_samples = len(self.trainset)
        self.replay_trainset = {f'task {item[0]}': None for item in self.taskcla}

        self.local_model = copy.deepcopy(model).to(self.device)
        self.loss = nn.CrossEntropyLoss()

        self.local_epochs = args.local_epochs
        self.replay_local_epochs = args.replay_local_epochs
        self.batch_size = args.batch_size
        self.replay_batch_size = args.replay_batch_size

        self.optimizer = None
        self.momentum = args.momentum
        self.lr = args.lr
        self.continual_lr = args.continual_lr
        self.replay_lr = args.replay_lr
        self.cur_lr = self.lr

        self.lr_scheduler = None
        self.warmup = args.warmup
        self.StepLR_step_size = args.step_size
        self.StepLR_gamma = args.gamma
        self.CosineAnnealingLR_T_max = args.T_max
        self.CosineAnnealingLR_replay_T_max = args.replay_T_max

        # 记忆的大小
        # SNN的时间步
        self.memory_size = args.memory_size
        self.timesteps = args.timesteps

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.local_model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.root_path = os.path.join(args.root_path, 'Client', f'id_{self.id}')
        self.logs_path = os.path.join(self.root_path, 'logs')
        self.models_path = os.path.join(self.root_path, 'models')

    def set_parameters(self, **kwargs):
        """
        根据相关参数设置本地相关参数
        Args:
            **kwargs: 相关参数

        Returns:

        """
        pass

    def set_optimizer(self, task_id: int, replay: bool):
        """
        根据任务的id、实验的名称和是否重播来设置优化器
        Args:
            task_id: 任务的id
            replay: 是否重放

        Returns:

        """
        # 获取本地模型参数（除了hlop层和神经元层） >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        params = []
        for name, param in self.local_model.named_parameters():
            if 'hlop' not in name:  # 如果该层不是hlop模块
                if task_id != 0:
                    if len(param.size()) != 1:
                        params.append(param)
                else:
                    params.append(param)

        if self.args.experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
            # 如果实验的名称是pmnist，设置replay=True才能真正重放
            if replay and self.args.experiment_name == 'pmnist':
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_lr)
                    self.cur_lr = self.replay_lr
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_lr)
                    self.cur_lr = self.replay_lr
                else:
                    raise NotImplementedError(self.args.opt)
            else:
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.lr)
                    self.cur_lr = self.lr
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.lr)
                    self.cur_lr = self.lr
                else:
                    raise NotImplementedError(self.args.opt)
        elif self.args.experiment_name == 'cifar':  # cifar 实验
            if replay:  # 如果重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_lr, momentum=self.momentum)
                    self.cur_lr = self.replay_lr
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_lr)
                    self.cur_lr = self.replay_lr
                else:
                    raise NotImplementedError(self.args.opt)
            else:  # 如果不重放
                if self.args.opt == 'SGD':
                    if task_id == 0:
                        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
                        self.cur_lr = self.lr
                    else:
                        self.optimizer = torch.optim.SGD(params, lr=self.continual_lr,
                                                         momentum=self.momentum)
                        self.cur_lr = self.continual_lr
                elif self.args.opt == 'Adam':
                    if task_id == 0:
                        self.optimizer = torch.optim.Adam(params, lr=self.lr)
                        self.cur_lr = self.lr
                    else:
                        self.optimizer = torch.optim.Adam(params, lr=self.continual_lr)
                        self.cur_lr = self.continual_lr
                else:
                    raise NotImplementedError(self.args.opt)
        elif self.args.experiment_name == 'miniimagenet':  # miniimagenet 实验
            if replay:  # 如果重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_lr, momentum=self.momentum)
                    self.cur_lr = self.replay_lr
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_lr)
                    self.cur_lr = self.replay_lr
                else:
                    raise NotImplementedError(self.args.opt)
            else:  # 如果不重放
                if self.args.opt == 'SGD':
                    if task_id == 0:
                        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
                        self.cur_lr = self.lr
                    else:
                        self.optimizer = torch.optim.SGD(params, lr=self.continual_lr,
                                                         momentum=self.momentum)
                        self.cur_lr = self.continual_lr
                elif self.args.opt == 'Adam':
                    if task_id == 0:
                        self.optimizer = torch.optim.Adam(params, lr=self.lr)
                        self.cur_lr = self.lr
                    else:
                        self.optimizer = torch.optim.Adam(params, lr=self.continual_lr)
                        self.cur_lr = self.continual_lr
                else:
                    raise NotImplementedError(self.args.opt)
        elif self.args.experiment_name == 'tinyimagenet':  # tiny-imagenet 实验
            if replay:  # 如果重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_lr, momentum=self.momentum)
                    self.cur_lr = self.replay_lr
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_lr)
                    self.cur_lr = self.replay_lr
                else:
                    raise NotImplementedError(self.args.opt)
            else:  # 如果不重放
                if self.args.opt == 'SGD':
                    if task_id == 0:
                        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
                        self.cur_lr = self.lr
                    else:
                        self.optimizer = torch.optim.SGD(params, lr=self.continual_lr,
                                                         momentum=self.momentum)
                        self.cur_lr = self.continual_lr
                elif self.args.opt == 'Adam':
                    if task_id == 0:
                        self.optimizer = torch.optim.Adam(params, lr=self.lr)
                        self.cur_lr = self.lr
                    else:
                        self.optimizer = torch.optim.Adam(params, lr=self.continual_lr)
                        self.cur_lr = self.continual_lr
                else:
                    raise NotImplementedError(self.args.opt)
        elif self.args.experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
            if replay:  # 如果重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_lr, momentum=self.momentum)
                    self.cur_lr = self.replay_lr
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_lr)
                    self.cur_lr = self.replay_lr
                else:
                    raise NotImplementedError(self.args.opt)
            else:  # 如果不重放
                if self.args.opt == 'SGD':
                    if task_id == 0:
                        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
                        self.cur_lr = self.lr
                    else:
                        self.optimizer = torch.optim.SGD(params, lr=self.continual_lr,
                                                         momentum=self.momentum)
                        self.cur_lr = self.continual_lr
                elif self.args.opt == 'Adam':
                    if task_id == 0:
                        self.optimizer = torch.optim.Adam(params, lr=self.lr)
                        self.cur_lr = self.lr
                    else:
                        self.optimizer = torch.optim.Adam(params, lr=self.continual_lr)
                        self.cur_lr = self.continual_lr
                else:
                    raise NotImplementedError(self.args.opt)
        elif self.args.experiment_name == 'svhn':  # svhn 实验
            if replay:  # 如果重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_lr, momentum=self.momentum)
                    self.cur_lr = self.replay_lr
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_lr)
                    self.cur_lr = self.replay_lr
                else:
                    raise NotImplementedError(self.args.opt)
            else:  # 如果不重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
                    self.cur_lr = self.lr
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.lr)
                    self.cur_lr = self.lr
                else:
                    raise NotImplementedError(self.args.opt)

    def set_learning_rate_scheduler(self, replay: bool):
        """
        根据实验的名称和是否重播来设置优化器
        Args:
            replay: 是否重放

        Returns:

        """
        if self.args.experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
            # 如果实验的名称是pmnist，设置replay=True才能真正重放
            if replay and self.args.experiment_name == 'pmnist':
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                   T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos(
                        (cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                          lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif self.args.experiment_name == 'cifar':  # cifar 实验
            if replay:
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                   T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos(
                        (cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                          lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif self.args.experiment_name == 'miniimagenet':  # miniimagenet 实验
            if replay:
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                   T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos(
                        (cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                          lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif self.args.experiment_name == 'tinyimagenet':  # tiny-imagenet 实验
            if self.optimizer is None:
                raise ValueError("Optimizer has not been initialized yet. Please call set_optimizer first.")
    
            if replay:
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                   T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos(
                        (cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                          lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif self.args.experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
            if replay:
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                   T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos(
                        (cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                          lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif self.args.experiment_name == 'svhn':  # svhn 实验
            if replay:
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                   T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                        step_size=self.StepLR_step_size,
                                                                        gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos(
                        (cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                          lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)

    def set_replay_data(self, task_id, n_task_class):
        replay_data, replay_label = [], []
        trainset = self.trainset[f'task {task_id}']
        data_of_classes = {key: torch.where(torch.tensor(trainset.labels.numpy() == key))[0] for key in
                           range(n_task_class)}

        for i in range(n_task_class):
            if len(data_of_classes[i]) == 0:
                continue
            elif len(data_of_classes[i]) < self.args.memory_size:
                replay_data.extend(trainset.data[data_of_classes[i]])
                replay_label.extend(trainset.labels[data_of_classes[i]])
            else:
                replay_data.extend(trainset.data[data_of_classes[i][:self.args.memory_size]])
                replay_label.extend(trainset.labels[data_of_classes[i][:self.args.memory_size]])

        # for i in range(n_task_class):
        #     n = self.args.memory_size
        #     idx = 0
        #     while n > 0:
        #         if trainset.labels[idx].numpy() == i:
        #             replay_data.append(trainset.data[idx])
        #             replay_label.append(trainset.labels[idx])
        #             n -= 1
        #         idx += 1

        replay_data = torch.stack(replay_data, dim=0)
        replay_label = torch.stack(replay_label, dim=0)

        self.replay_trainset[f'task {task_id}'] = GeneralDataset(data=replay_data, labels=replay_label, n_class=n_task_class)

    def train(self, task_id: int, bptt: bool, ottt: bool):
        """
        训练过程
        Args:
            task_id: 训练任务id
            bptt: 是否为BPTT训练方式
            ottt: 是否为OTTT训练方式

        Returns:

        """
        pass

    def replay(self, tasks_learned: list):
        """
        重放过程
        Args:
            tasks_learned: 需要重放的任务id的列表

        Returns:

        """
        pass

    def save_local_model(self, model_name):
        """
        保存本地模型
        Args:
            model_name: 模型名称（不需要绝对/相对路径）

        Returns:

        """
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        torch.save(self.local_model, os.path.join(self.models_path, f'{model_name}.pt'))

    def load_local_model(self, model_name):
        """
        加载本地模型
        Args:
            model_name:

        Returns:

        """
        model_abs_path = os.path.join(self.models_path, model_name)
        assert os.path.exists(model_abs_path)
        self.local_model = torch.load(model_abs_path)
