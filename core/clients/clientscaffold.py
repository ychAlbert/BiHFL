#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : SCAFFOLD算法的客户端类
import copy
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from progress.bar import Bar
from torch.utils.data import DataLoader

from core.clients.clientbase import Client
from core.utils import AverageMeter, accuracy, reset_net

__all__ = ['clientSCAFFOLD']


class clientSCAFFOLD(Client):
    def __init__(self, args, id, trainset, taskcla, model):
        super().__init__(args, id, trainset, taskcla, model)
        self.client_c = None  # 本地控制参数
        self.info_map = None
        self.global_c = None  # 全局控制参数
        self.global_model = None  # 全局模型

    def set_parameters(self, model, global_c):
        """
        从服务器接收到全局的模型和控制参数更新本地的模型和控制参数
        Args:
            model: 全局的模型
            global_c: 全局的控制参数

        Returns:

        """
        # 利用服务器接收到的全局模型更新本地模型
        for param_old, param_new in zip(self.local_model.parameters(), model.parameters()):
            param_old.data = param_new.data.clone()
        # 获取全局控制参数和全局模型
        self.global_c = global_c
        self.global_model = model

    def update_c_before_train(self):
        # 获取当前本地模型
        current_local_model = copy.deepcopy(self.local_model)

        # 通过有序字典存放模型（层名称：参数）
        info_map_new = OrderedDict()
        client_c_new = []
        start_index = 0
        for name, param in current_local_model.named_parameters():
            param_num = len(param.view(-1))
            info_map_new[name] = param_num
            client_c_new.append(torch.zeros_like(param))
            start_index += param_num

        if self.client_c is not None:
            layers = self.info_map.keys()
            for idx, (name, param) in enumerate(current_local_model.named_parameters()):
                if name in layers:
                    # 如果对应层的参数数量相等，那么就直接替换
                    if info_map_new[name] == self.info_map[name]:
                        client_c_new[idx] = self.client_c[idx]
                    # 如果对应层的新参数数量大于旧参数数量，那么就进行填充
                    if info_map_new[name] > self.info_map[name] and name.startswith('hlop'):
                        param_temp_new, param_temp_old = client_c_new[idx], self.client_c[idx]
                        m, n = param_temp_new.shape[0], param_temp_new.shape[1]
                        a, b = param_temp_old.shape[0], param_temp_old.shape[1]
                        padding_row = m - a
                        padding_col = n - b
                        param_temp = F.pad(param_temp_old, (0, padding_col, 0, padding_row))
                        client_c_new[idx] = param_temp
                    elif info_map_new[name] < self.info_map[name] and name.startswith('hlop'):
                        param_temp_new, param_temp_old = client_c_new[idx], self.client_c[idx]
                        m, n = param_temp_new.shape[0], param_temp_new.shape[1]
                        param_temp = param_temp_old[:m, :n]
                        client_c_new[idx] = param_temp

        self.client_c = client_c_new
        self.info_map = info_map_new

    def update_c_after_train(self):
        """
        更新控制参数, 对应论文算法1的第12行中的第二种方式(常用)
        @return:
        """
        # local_model = self.local_model.state_dict()
        # global_controls = self.controls_global.state_dict()
        # global_model = self.global_model.state_dict()
        #
        # local_controls = self.controls_local.state_dict()
        # for param in local_controls:
        #     local_controls[param] = local_controls[param] - global_controls[param] + 1 / (
        #             self.local_epochs * self.cur_lr) * (global_model[param] - local_model[param])
        # self.controls_local.load_state_dict(local_controls)

        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(),
                                self.local_model.parameters()):
            ci.data = ci - c + 1 / (self.local_epochs * self.cur_lr) * (x - yi)

    def delta_yc(self):
        """
        计算模型参数和控制变量的变化量, 对应论文算法1的第13行
        @return:
        """
        # global_controls = self.controls_global.state_dict()
        # global_model = self.global_model.state_dict()
        # local_model = self.local_model.state_dict()
        #
        # delta_model = copy.deepcopy(local_model)  # 模型参数的变化量
        # delta_controls = copy.deepcopy(local_model)  # 控制参数的变化量
        # for param in delta_model:
        #     delta_model[param] = 0.0
        #     delta_controls[param] = 0.0
        # for param in local_model:
        #     delta_model[param] = local_model[param] - global_model[param]
        #     delta_controls[param] = -global_controls[param] + 1 / (self.local_epochs * self.cur_lr) * (
        #             global_model[param] - local_model[param])
        # return delta_model, delta_controls
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.local_model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1 / (self.local_epochs * self.cur_lr) * (x - yi))

        return delta_y, delta_c

    def train(self, task_id: int, bptt: bool, ottt: bool):
        # --------------------------------------------------------------------------------------------------------------
        # 数据集相关内容
        # --------------------------------------------------------------------------------------------------------------
        trainset = self.trainset[f'task {task_id}']
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        n_trainset = len(trainset)

        # --------------------------------------------------------------------------------------------------------------
        # 进度条相关指标及设置
        # --------------------------------------------------------------------------------------------------------------
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Client {:^3d} Training'.format(self.id), max=((n_trainset - 1) // self.batch_size + 1))

        # --------------------------------------------------------------------------------------------------------------
        # 训练主体部分
        # --------------------------------------------------------------------------------------------------------------
        # 开启模型训练模式
        self.local_model.train()

        if task_id != 0:
            self.local_model.fix_bn()

        batch_idx = 0

        self.update_c_before_train()  # 更新控制参数
        # 本地轮次的操作
        for local_epoch in range(1, self.local_epochs + 1):
            for data, label in trainloader:
                data = data.to(self.device)
                label = label.to(self.device)

                if ottt:
                    total_loss = 0.
                    if not self.args.online_update:
                        self.optimizer.zero_grad()
                    for t in range(self.timesteps):
                        if self.args.online_update:
                            self.optimizer.zero_grad()
                        init = (t == 0)

                        flag = self.args.use_hlop and (local_epoch > self.args.hlop_start_epochs)
                        if task_id == 0:
                            out_fr_, out_fr = self.local_model(data, task_id, projection=False, update_hlop=flag,
                                                               init=init)
                        else:
                            out_fr_, out_fr = self.local_model(data, task_id, projection=self.args.use_hlop,
                                                               proj_id_list=[0], update_hlop=flag,
                                                               fix_subspace_id_list=[0], init=init)

                        if t == 0:
                            total_fr = out_fr.clone().detach()
                        else:
                            total_fr += out_fr.clone().detach()
                        loss = self.loss(out_fr, label) / self.timesteps
                        loss.backward()
                        total_loss += loss.detach()
                        if self.args.online_update:
                            self.optimizer.step()
                    if not self.args.online_update:
                        self.optimizer.step()

                    out = total_fr
                elif bptt:
                    self.optimizer.zero_grad()

                    flag = self.args.use_hlop and (local_epoch > self.args.hlop_start_epochs)
                    if task_id == 0:
                        out_, out = self.local_model(data, task_id, projection=False, update_hlop=flag)
                    else:
                        out_, out = self.local_model(data, task_id, projection=self.args.use_hlop, proj_id_list=[0],
                                                     update_hlop=flag, fix_subspace_id_list=[0])
                    loss = self.loss(out, label)
                    loss.backward()
                    self.optimizer.step()

                    reset_net(self.local_model)
                else:
                    data = data.unsqueeze(1)
                    data = data.repeat(1, self.timesteps, 1, 1, 1)
                    self.optimizer.zero_grad()

                    flag = self.args.use_hlop and (local_epoch > self.args.hlop_start_epochs)
                    if task_id == 0:
                        out_, out = self.local_model(data, task_id, projection=False, update_hlop=flag)
                    else:
                        out_, out = self.local_model(data, task_id, projection=self.args.use_hlop, proj_id_list=[0],
                                                     update_hlop=flag, fix_subspace_id_list=[0])
                    loss = self.loss(out, label)
                    loss.backward()
                    self.optimizer.step()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                losses.update(loss, data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx,
                    size=((n_trainset - 1) // self.batch_size + 1) * self.local_epochs,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
                bar.next()
        bar.finish()

        self.lr_scheduler.step()  # 学习率调度器更新
        self.update_c_after_train()  # 更新控制参数

    def replay(self, tasks_learned):
        self.local_model.train()
        self.local_model.fix_bn()

        for replay_task in tasks_learned:
            replay_trainset = self.replay_trainset[f'task {replay_task}']
            replay_trainloader = DataLoader(replay_trainset, batch_size=self.replay_batch_size, shuffle=True,
                                            drop_last=False)
            n_replay_trainset = len(replay_trainset)

            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            end = time.time()
            bar = Bar('Client {:^3d} Replaying Task {:^2d}'.format(self.id, replay_task),
                      max=((n_replay_trainset - 1) // self.replay_batch_size + 1))

            n_replay_traindata = 0
            train_acc = 0
            train_loss = 0
            batch_idx = 0

            global_controls = self.global_c.state_dict()
            local_controls = self.client_c.state_dict()

            for epoch in range(1, self.replay_local_epochs + 1):
                for data, label in replay_trainloader:
                    data = data.to(self.device)
                    label = label.to(self.device)

                    self.optimizer.zero_grad()

                    data = data.unsqueeze(1)
                    data = data.repeat(1, self.timesteps, 1, 1, 1)
                    rep, out = self.local_model(data, replay_task, projection=False, update_hlop=False)

                    # 计算loss
                    loss = self.loss(out, label)
                    loss.backward()
                    self.optimizer.step()

                    # 更新本地模型参数
                    local_model = self.local_model.state_dict()
                    for param in local_model:
                        local_model[param] = local_model[param] - self.cur_lr * (
                                global_controls[param] - local_controls[param])
                        self.local_model.load_state_dict(local_model)

                    train_loss += loss.item() * label.numel()

                    prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                    losses.update(loss, data.size(0))
                    top1.update(prec1.item(), data.size(0))
                    top5.update(prec5.item(), data.size(0))
                    n_replay_traindata += label.numel()
                    train_acc += (out.argmax(1) == label).float().sum().item()

                    batch_time.update(time.time() - end)
                    end = time.time()
                    batch_idx += 1

                    bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx,
                        size=((n_replay_trainset - 1) // self.replay_batch_size + 1) * self.replay_local_epochs,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    )
                    bar.next()
            bar.finish()
            self.lr_scheduler.step()
            self.update_c_after_train()  # 更新控制参数
