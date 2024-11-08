#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : SCAFFOLD算法的客户端类
import copy
import time

import numpy as np
import torch
from progress.bar import Bar
from torch.utils.data import DataLoader

from ..utils import AverageMeter
from ..clients.clientbase import Client
from ..utils import accuracy, reset_net

__all__ = ['clientSCAFFOLD']


class clientSCAFFOLD(Client):
    def __init__(self, args, id, trainset, local_model, taskcla, **kwargs):
        super().__init__(args, id, trainset, local_model, taskcla, **kwargs)
        # 本地控制参数
        self.local_controls = None
        # 全局控制参数
        self.global_controls = None
        # 全局模型
        self.global_model = None

        self.num_batches = None
        self.max_local_epochs = None

    def set_parameters(self, global_model, global_controls):
        """
        从服务器接收到全局的模型和控制参数更新本地的模型和控制参数
        @param global_model: 全局的模型
        @param global_controls: 全局的控制参数
        @return:
        """
        # 利用服务器接收到的全局模型更新本地模型
        for gp, lp in zip(global_model.parameters(), self.local_model.parameters()):
            lp.data = gp.data.clone()
        # 获取全局的控制参数
        self.global_controls = copy.deepcopy(global_controls)
        # 获取全局的模型
        self.global_model = copy.deepcopy(global_model)

    def update_yc(self, task_id):
        trainset = self.trainset[f'task {task_id}']
        num_batches = np.ceil(len(trainset) // self.batch_size)
        for local_param, local_control, global_param, global_control in zip(self.local_model.parameters(),
                                                                            self.local_controls,
                                                                            self.global_model.parameters(),
                                                                            self.global_controls):
            local_control.data = local_control - global_control + (
                    global_param - local_param) / num_batches / self.max_local_epochs / self.current_learning_rate

    def delta_yc(self, task_id):
        """
        计算模型和控制变量的变化值
        @param task_id:
        @return:
        """
        delta_model, delta_control = [], []
        trainset = self.trainset[f'task {task_id}']
        num_batches = np.ceil(len(trainset) // self.batch_size)
        for c, x, yi in zip(self.global_controls, self.global_model.parameters(), self.local_model.parameters()):
            delta_model.append(yi - x)
            delta_control.append(- c + 1 / num_batches / self.max_local_epochs / self.current_learning_rate * (x - yi))

        return delta_model, delta_control

    def train(self, task_id):
        # --------------------------------------------------------------------------------------------------------------
        # 获取实验相关参数（是否是bptt/ottt相关设置）
        # --------------------------------------------------------------------------------------------------------------
        bptt, ottt = False, False
        if self.args.experiment_name.endswith('bptt'):
            bptt = True
        elif self.args.experiment_name.endswith('ottt'):
            ottt = True

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
        self.local_controls = [torch.zeros_like(param) for param in self.local_model.parameters()]

        if task_id != 0:
            self.local_model.fix_bn()

        n_traindata = 0
        train_acc = 0
        train_loss = 0
        batch_idx = 0

        start_time = time.time()
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
                            out_fr_, out_fr = self.local_model(data, task_id, projection=False, update_hlop=flag, init=init)
                        else:
                            out_fr_, out_fr = self.local_model(data, task_id, projection=self.args.use_hlop, proj_id_list=[0], update_hlop=flag, fix_subspace_id_list=[0], init=init)

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
                    for lp, gc, lc in zip(self.local_model.parameters(), self.global_controls, self.local_controls):
                        if lp.grad is None:
                            lp.data = lp.data - self.current_learning_rate * (gc.data.clone() - lc.data.clone())
                        else:
                            lp.data = lp.data - self.current_learning_rate * (
                                    lp.grad.data + gc.data.clone() - lc.data.clone())
                    train_loss += total_loss.item() * label.numel()
                    out = total_fr
                elif bptt:
                    self.optimizer.zero_grad()

                    flag = self.args.use_hlop and (local_epoch > self.args.hlop_start_epochs)
                    if task_id == 0:
                        out_, out = self.local_model(data, task_id, projection=False, update_hlop=flag)
                    else:
                        out_, out = self.local_model(data, task_id, projection=self.args.use_hlop, proj_id_list=[0], update_hlop=flag, fix_subspace_id_list=[0])
                    loss = self.loss(out, label)
                    loss.backward()
                    self.optimizer.step()

                    for lp, gc, lc in zip(self.local_model.parameters(), self.global_controls, self.local_controls):
                        if lp.grad is None:
                            lp.data = lp.data - self.current_learning_rate * (gc.data.clone() - lc.data.clone())
                        else:
                            lp.data = lp.data - self.current_learning_rate * (
                                    lp.grad.data + gc.data.clone() - lc.data.clone())
                    reset_net(self.local_model)
                    train_loss += loss.item() * label.numel()
                else:
                    data = data.unsqueeze(1)
                    data = data.repeat(1, self.timesteps, 1, 1, 1)
                    self.optimizer.zero_grad()

                    flag = self.args.use_hlop and (local_epoch > self.args.hlop_start_epochs)
                    if task_id == 0:
                        out_, out = self.local_model(data, task_id, projection=False, update_hlop=flag)
                    else:
                        out_, out = self.local_model(data, task_id, projection=self.args.use_hlop, proj_id_list=[0], update_hlop=flag, fix_subspace_id_list=[0])
                    loss = self.loss(out, label)
                    loss.backward()
                    self.optimizer.step()

                    for lp, gc, lc in zip(self.local_model.parameters(), self.global_controls, self.local_controls):
                        if lp.grad is None:
                            lp.data = lp.data - self.current_learning_rate * (gc.data.clone() - lc.data.clone())
                        else:
                            lp.data = lp.data - self.current_learning_rate * (lp.grad.data + gc.data.clone() - lc.data.clone())

                    train_loss += loss.item() * label.numel()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                losses.update(loss, data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))

                n_traindata += label.numel()
                train_acc += (out.argmax(1) == label).float().sum().item()

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

        train_loss /= n_traindata
        train_acc /= n_traindata
        self.learning_rate_scheduler.step()
        self.update_yc(task_id)

        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['num_rounds'] += 1

    def replay(self, tasks_learned):
        self.local_model.train()
        self.local_model.fix_bn()

        for replay_task in tasks_learned:
            replay_trainset = self.replay_trainset[f'task {replay_task}']
            replay_trainloader = DataLoader(replay_trainset, batch_size=self.replay_batch_size, shuffle=True, drop_last=False)
            n_replay_trainset = len(replay_trainset)

            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            end = time.time()
            bar = Bar('Client {:^3d} Replaying Task {:^2d}'.format(self.id, replay_task), max=((n_replay_trainset - 1) // self.replay_batch_size + 1))

            n_replay_traindata = 0
            train_acc = 0
            train_loss = 0
            batch_idx = 0

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

                    for lp, gc, lc in zip(self.local_model.parameters(), self.global_controls, self.local_controls):
                        if lp.grad is None:
                            lp.data = lp.data - self.current_learning_rate * (gc.data.clone() - lc.data.clone())
                        else:
                            lp.data = lp.data - self.current_learning_rate * (lp.grad.data + gc.data.clone() - lc.data.clone())

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
            self.learning_rate_scheduler.step()
