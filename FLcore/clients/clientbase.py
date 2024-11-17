#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : 联邦学习客户端的基础类
import copy
import math
import os

import torch
import torch.nn as nn

__all__ = ['Client']

from FLcore.utils import GeneralDataset


class Client(object):
    def __init__(self, args, id, trainset, model, taskcla):
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
        self.learning_rate = args.learning_rate
        self.continual_learning_rate = args.continual_learning_rate
        self.replay_learning_rate = args.replay_learning_rate
        self.current_learning_rate = self.learning_rate

        self.learning_rate_scheduler = None
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

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.root_path = os.path.join(args.root_path, 'Client', f'id_{self.id}')
        self.logs_path = os.path.join(self.root_path, 'logs')
        self.models_path = os.path.join(self.root_path, 'models')

    def set_parameters(self, model):
        """
        根据接收到的模型参数设置本地模型参数
        :param model: 接收到的模型
        :return:
        """
        pass

    def set_optimizer(self, task_id: int, for_replaying: bool):
        """
        根据任务的id、实验的名称和是否重播来设置优化器
        @param task_id: 任务的id
        @param for_replaying: 为了重播
        @return:
        """
        # 获取本地模型参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        params = []
        for name, p in self.local_model.named_parameters():
            if 'hlop' not in name:
                if task_id != 0:
                    if len(p.size()) != 1:
                        params.append(p)
                else:
                    params.append(p)
        # 获取本地模型参数 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if self.args.experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
            # 如果实验的名称是pmnist，设置replay=True才能真正重放
            if for_replaying and self.args.experiment_name == 'pmnist':
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_learning_rate)
                    self.current_learning_rate = self.replay_learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_learning_rate)
                    self.current_learning_rate = self.replay_learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
            else:
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.learning_rate)
                    self.current_learning_rate = self.learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                    self.current_learning_rate = self.learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
        elif self.args.experiment_name == 'cifar':  # cifar 实验
            if for_replaying:  # 如果重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_learning_rate, momentum=self.momentum)
                    self.current_learning_rate = self.replay_learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_learning_rate)
                    self.current_learning_rate = self.replay_learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
            else:  # 如果不重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum)
                    self.current_learning_rate = self.learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                    self.current_learning_rate = self.learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
        elif self.args.experiment_name == 'miniimagenet':  # miniimagenet 实验
            if for_replaying:
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_learning_rate, momentum=self.momentum)
                    self.current_learning_rate = self.replay_learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_learning_rate)
                    self.current_learning_rate = self.replay_learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
            else:
                if self.args.opt == 'SGD':
                    if task_id == 0:
                        self.optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum)
                        self.current_learning_rate = self.learning_rate
                    else:
                        self.optimizer = torch.optim.SGD(params, lr=self.continual_learning_rate,
                                                         momentum=self.momentum)
                        self.current_learning_rate = self.continual_learning_rate
                elif self.args.opt == 'Adam':
                    if task_id == 0:
                        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                        self.current_learning_rate = self.learning_rate
                    else:
                        self.optimizer = torch.optim.Adam(params, lr=self.continual_learning_rate)
                        self.current_learning_rate = self.continual_learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
        elif self.args.experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
            if for_replaying:  # 如果重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_learning_rate, momentum=self.momentum)
                    self.current_learning_rate = self.replay_learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_learning_rate)
                    self.current_learning_rate = self.replay_learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
            else:  # 如果不重放
                if self.args.opt == 'SGD':
                    if task_id == 0:
                        self.optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum)
                        self.current_learning_rate = self.learning_rate
                    else:
                        self.optimizer = torch.optim.SGD(params, lr=self.continual_learning_rate,
                                                         momentum=self.momentum)
                        self.current_learning_rate = self.continual_learning_rate
                elif self.args.opt == 'Adam':
                    if task_id == 0:
                        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                        self.current_learning_rate = self.learning_rate
                    else:
                        self.optimizer = torch.optim.Adam(params, lr=self.continual_learning_rate)
                        self.current_learning_rate = self.continual_learning_rate
                else:
                    raise NotImplementedError(self.args.opt)

    def set_learning_rate_scheduler(self, for_replaying: bool):
        """
        根据实验的名称和是否重播来设置优化器
        @param for_replaying: 为了重播
        @return:
        """
        if self.args.experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
            # 如果实验的名称是pmnist，设置replay=True才能真正重放
            if for_replaying and self.args.experiment_name == 'pmnist':
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos((cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                                     lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif self.args.experiment_name == 'cifar':  # cifar 实验
            if for_replaying:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos((cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                                     lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif self.args.experiment_name == 'miniimagenet':  # miniimagenet 实验
            if for_replaying:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos((cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                                     lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif self.args.experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
            if for_replaying:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos((cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                                     lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)

    def set_replay_data(self, task_id, n_task_class):
        replay_data, replay_label = [], []
        trainset = self.trainset[f'task {task_id}']
        data_of_classes = {key: torch.where(torch.tensor(trainset.labels.numpy() == key))[0] for key in range(n_task_class)}

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

    def train(self, task_id):
        pass

    def replay(self, tasks_learned):
        pass

    def save_local_model(self, model_name):
        """
        保存本地模型
        :param model_name: 模型名称（不需要绝对/相对路径）
        :return:
        """
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        torch.save(self.local_model, os.path.join(self.models_path, f'{model_name}.pt'))

    def load_local_model(self, model_name):
        """
        加载本地模型
        :param model_name: 模型名称（不需要绝对/相对路径）
        :return:
        """
        model_abs_path = os.path.join(self.models_path, model_name)
        assert os.path.exists(model_abs_path)
        self.local_model = torch.load(model_abs_path)
