#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : SCAFFOLD算法的客户端类
import copy
import time

import numpy as np
import torch
from progress.bar import Bar

from ..utils import AverageMeter
from ..clients.clientbase import Client
from ..utils import accuracy
from ..utils.model_utils import reset_net

__all__ = ['clientSCAFFOLD']


class clientSCAFFOLD(Client):
    def __init__(self, args, id, xtrain, ytrain, local_model, **kwargs):
        super().__init__(args, id, xtrain, ytrain, local_model, **kwargs)
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
        num_batches = np.ceil(len(self.xtrain[task_id]) // self.batch_size)
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

        num_batches = np.ceil(len(self.xtrain[task_id]) // self.batch_size)
        for c, x, yi in zip(self.global_controls, self.global_model.parameters(), self.local_model.parameters()):
            delta_model.append(yi - x)
            delta_control.append(- c + 1 / num_batches / self.max_local_epochs / self.current_learning_rate * (x - yi))

        return delta_model, delta_control

    def train(self, task_id, HLOP_SNN):
        # --------------------------------------------------------------------------------------------------------------
        # 获取实验相关参数（是否是HLOP_SNN相关实验，如果是的话，是否是bptt/ottt相关设置）
        # --------------------------------------------------------------------------------------------------------------
        bptt, ottt = False, False
        if HLOP_SNN:
            if self.experiment_name.endswith('bptt'):
                bptt = True
            elif self.experiment_name.endswith('ottt'):
                ottt = True

        # --------------------------------------------------------------------------------------------------------------
        # 获取本地的epoch
        # --------------------------------------------------------------------------------------------------------------
        self.max_local_epochs = self.local_epochs
        if self.train_slow:
            self.max_local_epochs = np.random.randint(1, self.max_local_epochs // 2)

        # --------------------------------------------------------------------------------------------------------------
        # 数据集相关内容
        # --------------------------------------------------------------------------------------------------------------
        # 对应任务的训练集
        xtrain = self.xtrain[task_id]
        # 对应任务的测试集
        ytrain = self.ytrain[task_id]
        # 获取对应任务的数据集数量
        num_trainset = len(xtrain)

        # --------------------------------------------------------------------------------------------------------------
        # 进度条相关指标及设置
        # --------------------------------------------------------------------------------------------------------------
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Client {:^3d} Training'.format(self.id), max=((num_trainset - 1) // self.batch_size + 1))

        # --------------------------------------------------------------------------------------------------------------
        # 训练主体部分
        # --------------------------------------------------------------------------------------------------------------
        # 开启模型训练模式
        self.local_model.train()
        self.local_controls = [torch.zeros_like(param) for param in self.local_model.parameters()]

        if task_id != 0 and HLOP_SNN:
            self.local_model.fix_bn()

        train_num = 0
        train_acc = 0
        train_loss = 0
        batch_idx = 0

        samples_index = np.arange(xtrain.size(0))
        np.random.shuffle(samples_index)

        start_time = time.time()
        # 本地轮次的操作
        for local_epoch in range(1, self.local_epochs + 1):
            # 一个轮次中的批处理操作
            for i in range(0, len(samples_index), self.batch_size):
                # 如果可以获取完整的批次，那么就获取完整批次
                if i + self.batch_size <= len(samples_index):
                    index = samples_index[i: i + self.batch_size]
                # 如果没有完整的批次可供获取，那么获取所有剩下的
                else:
                    index = samples_index[i:]
                batch_idx += 1

                # 获取一个批次的数据并转化为Tensor
                data = xtrain[index].float().to(self.device)
                # 获取一个批次的标签并转化为Tensor
                target = ytrain[index].to(self.device)

                if ottt:
                    total_loss = 0.
                    if not self.args.online_update:
                        self.optimizer.zero_grad()
                    for t in range(self.timesteps):
                        if self.args.online_update:
                            self.optimizer.zero_grad()
                        init = (t == 0)
                        if task_id == 0:
                            flag = not (self.args.baseline and (local_epoch <= self.args.hlop_start_epochs))
                            out_fr_, out_fr = self.local_model(data, task_id, projection=False, update_hlop=flag, init=init)
                        else:
                            flag = not (self.args.baseline or (local_epoch <= self.args.hlop_start_epochs))
                            out_fr_, out_fr = self.local_model(data, task_id, projection=not self.args.baseline, proj_id_list=[0],
                                                               update_hlop=flag, fix_subspace_id_list=[0], init=init)
                        if t == 0:
                            total_fr = out_fr.clone().detach()
                        else:
                            total_fr += out_fr.clone().detach()
                        loss = self.loss(out_fr, target) / self.timesteps
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
                    train_loss += total_loss.item() * target.numel()
                    out = total_fr
                elif bptt:
                    self.optimizer.zero_grad()
                    if task_id == 0:
                        flag = not (self.args.baseline and (local_epoch <= self.args.hlop_start_epochs))
                        out_, out = self.local_model(data, task_id, projection=False, update_hlop=flag)
                    else:
                        flag = not (self.args.baseline or (local_epoch <= self.args.hlop_start_epochs))
                        out_, out = self.local_model(data, task_id, projection=not self.args.baseline, proj_id_list=[0],
                                                     update_hlop=flag, fix_subspace_id_list=[0])
                    loss = self.loss(out, target)
                    loss.backward()
                    self.optimizer.step()

                    for lp, gc, lc in zip(self.local_model.parameters(), self.global_controls, self.local_controls):
                        if lp.grad is None:
                            lp.data = lp.data - self.current_learning_rate * (gc.data.clone() - lc.data.clone())
                        else:
                            lp.data = lp.data - self.current_learning_rate * (
                                    lp.grad.data + gc.data.clone() - lc.data.clone())
                    reset_net(self.local_model)
                    train_loss += loss.item() * target.numel()
                else:
                    data = data.unsqueeze(1)
                    data = data.repeat(1, self.timesteps, 1, 1, 1)
                    self.optimizer.zero_grad()

                    if task_id == 0:
                        flag = not (self.args.baseline and (local_epoch <= self.args.hlop_start_epochs))
                        out_, out = self.local_model(data, task_id, projection=False, update_hlop=flag)

                    else:
                        flag = not (self.args.baseline or (local_epoch <= self.args.hlop_start_epochs))
                        out_, out = self.local_model(data, task_id, projection=not self.args.baseline, proj_id_list=[0],
                                                     update_hlop=flag, fix_subspace_id_list=[0])
                    loss = self.loss(out, target)
                    loss.backward()
                    self.optimizer.step()

                    for lp, gc, lc in zip(self.local_model.parameters(), self.global_controls, self.local_controls):
                        if lp.grad is None:
                            lp.data = lp.data - self.current_learning_rate * (gc.data.clone() - lc.data.clone())
                        else:
                            lp.data = lp.data - self.current_learning_rate * (lp.grad.data + gc.data.clone() - lc.data.clone())

                    train_loss += loss.item() * target.numel()


                # measure accuracy and record loss
                prec1, prec5 = accuracy(out.data, target.data, topk=(1, 5))
                losses.update(loss, data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))

                train_num += target.numel()
                train_acc += (out.argmax(1) == target).float().sum().item()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx,
                    size=((xtrain.size(0) - 1) // self.batch_size + 1) * self.local_epochs,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
                bar.next()
        bar.finish()

        train_loss /= train_num
        train_acc /= train_num
        self.learning_rate_scheduler.step()
        self.update_yc(task_id)

        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['num_rounds'] += 1

    def replay(self, tasks_learned, HLOP_SNN):
        self.local_model.train()
        self.local_model.fix_bn()

        for replay_task in tasks_learned:
            # 对应任务的训练集
            xtrain = self.replay_xtrain[replay_task]
            # 对应任务的测试集
            ytrain = self.replay_ytrain[replay_task]
            # 获取对应任务的数据集数量
            num_trainset = len(xtrain)

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            end = time.time()
            bar = Bar('Client {:^3d} Replaying Task {:^2d}'.format(self.id, replay_task),
                      max=((num_trainset - 1) // self.replay_batch_size + 1))

            task_data_num = len(xtrain)
            r = np.arange(task_data_num)
            np.random.shuffle(r)

            train_num = 0
            train_acc = 0
            train_loss = 0
            batch_idx = 0

            for epoch in range(1, self.replay_local_epochs + 1):
                for i in range(0, task_data_num, self.replay_batch_size):
                    if i + self.replay_batch_size <= task_data_num:
                        index = r[i: i + self.replay_batch_size]
                    else:
                        index = r[i:]
                    batch_idx += 1

                    # 获取一个批次的数据并转化为Tensor
                    data = xtrain[index].float().to(self.device)
                    # 获取一个批次的标签并转化为Tensor
                    target = ytrain[index].to(self.device)

                    self.optimizer.zero_grad()
                    if HLOP_SNN:
                        data = data.unsqueeze(1)
                        data = data.repeat(1, self.timesteps, 1, 1, 1)
                        rep, out = self.local_model(data, replay_task, projection=False, update_hlop=False)
                    else:
                        rep, out = self.local_model(data)

                    # 计算loss
                    loss = self.loss(out, target)
                    loss.backward()
                    self.optimizer.step()

                    for lp, gc, lc in zip(self.local_model.parameters(), self.global_controls, self.local_controls):
                        if lp.grad is None:
                            lp.data = lp.data - self.current_learning_rate * (gc.data.clone() - lc.data.clone())
                        else:
                            lp.data = lp.data - self.current_learning_rate * (lp.grad.data + gc.data.clone() - lc.data.clone())

                    train_loss += loss.item() * target.numel()

                    prec1, prec5 = accuracy(out.data, target.data, topk=(1, 5))
                    losses.update(loss, data.size(0))
                    top1.update(prec1.item(), data.size(0))
                    top5.update(prec5.item(), data.size(0))
                    train_num += target.numel()
                    train_acc += (out.argmax(1) == target).float().sum().item()

                    batch_time.update(time.time() - end)
                    end = time.time()

                    bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx,
                        size=((num_trainset - 1) // self.replay_batch_size + 1) * self.replay_local_epochs,
                        data=data_time.avg,
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
