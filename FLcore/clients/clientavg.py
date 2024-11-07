#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : FedAvg算法的客户端类
import time

import numpy as np
from progress.bar import Bar

from ..utils import AverageMeter
from ..clients.clientbase import Client
from ..utils import accuracy
from ..utils.model_utils import reset_net

__all__ = ['clientAVG']


class clientAVG(Client):
    def __init__(self, args, id, xtrain, ytrain, local_model, **kwargs):
        super().__init__(args, id, xtrain, ytrain, local_model, **kwargs)

    def set_parameters(self, model):
        """
        根据接收到的模型参数设置本地模型参数
        :param model: 接收到的模型
        :return:
        """
        for new_param, old_param in zip(model.parameters(), self.local_model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self, task_id):
        # --------------------------------------------------------------------------------------------------------------
        # 获取实验相关参数（是否是HLOP_SNN相关实验，如果是的话，是否是bptt/ottt相关设置）
        # --------------------------------------------------------------------------------------------------------------
        bptt, ottt = False, False
        if self.args.use_HLOP:
            if self.args.experiment_name.endswith('bptt'):
                bptt = True
            elif self.args.experiment_name.endswith('ottt'):
                ottt = True

        # --------------------------------------------------------------------------------------------------------------
        # 获取本地的epoch
        # --------------------------------------------------------------------------------------------------------------
        if self.train_slow:
            self.local_epochs = np.random.randint(1, self.local_epochs // 2)

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

        if task_id != 0 and self.args.use_HLOP:
            self.local_model.fix_bn()

        train_num = 0
        train_acc = 0
        train_loss = 0
        batch_idx = 0

        samples_index = np.arange(num_trainset)
        np.random.shuffle(samples_index)

        # 开始时间
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
                    total_loss = 0.0
                    if not self.args.online_update:
                        self.optimizer.zero_grad()
                    for t in range(self.timesteps):
                        if self.args.online_update:
                            self.optimizer.zero_grad()
                        init = (t == 0)
                        if task_id == 0:
                            flag = not (self.args.baseline and (local_epoch <= self.args.hlop_start_epochs))
                            out_fr_, out_fr = self.local_model(data, task_id, projection=False, update_hlop=flag,
                                                               init=init)
                        else:
                            flag = not (self.args.baseline or (local_epoch <= self.args.hlop_start_epochs))
                            out_fr_, out_fr = self.local_model(data, task_id, projection=not self.args.baseline,
                                                               proj_id_list=[0],
                                                               update_hlop=flag, fix_subspace_id_list=[0],
                                                               init=init)
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
                    train_loss += total_loss.item() * target.numel()
                    out = total_fr
                elif bptt:
                    self.optimizer.zero_grad()
                    if task_id == 0:
                        flag = not (self.args.baseline and (local_epoch <= self.args.hlop_start_epochs))
                        out_, out = self.local_model(data, task_id, projection=False, update_hlop=flag)
                    else:
                        flag = not (self.args.baseline or (local_epoch <= self.args.hlop_start_epochs))
                        out_, out = self.local_model(data, task_id, projection=not self.args.baseline,
                                                     proj_id_list=[0],
                                                     update_hlop=flag, fix_subspace_id_list=[0])
                    loss = self.loss(out, target)
                    loss.backward()
                    self.optimizer.step()
                    reset_net(self.local_model)
                    train_loss += loss.item() * target.numel()
                else:
                    data = data.unsqueeze(1)
                    data = data.repeat(1, self.timesteps, 1, 1, 1)
                    # ----------------------------------------------------------------------------------------------
                    # 核心网络训练过程
                    # ----------------------------------------------------------------------------------------------
                    # 清空参数梯度
                    self.optimizer.zero_grad()
                    # 模型推理
                    if task_id == 0:
                        flag = not (self.args.baseline and (local_epoch <= self.args.hlop_start_epochs))
                        if self.args.use_HLOP:
                            out_, out = self.local_model(data, task_id, projection=False, update_hlop=flag)
                        else:  # 可能有错
                            out_, out = self.local_model(data)
                    else:
                        flag = not (self.args.baseline or (local_epoch <= self.args.hlop_start_epochs))
                        if self.args.use_HLOP:
                            out_, out = self.local_model(data, task_id, projection=not self.args.baseline,
                                                         proj_id_list=[0],
                                                         update_hlop=flag, fix_subspace_id_list=[0])
                        else:  # 可能有错
                            out_, out = self.local_model(data)
                    # 计算loss
                    loss = self.loss(out, target)
                    # loss反向传播
                    loss.backward()
                    # 参数更新
                    self.optimizer.step()
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

                bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx,
                    size=((num_trainset - 1) // self.batch_size + 1) * self.local_epochs,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
                bar.next()
        bar.finish()
        self.save_local_model(str(time.time()))

        train_loss /= train_num
        train_acc /= train_num

        self.learning_rate_scheduler.step()

        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['num_rounds'] += 1

    def replay(self, tasks_learned):
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
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            end = time.time()
            bar = Bar('Client {:^3d} Replaying Task {:^2d}'.format(self.id, replay_task), max=((num_trainset - 1) // self.replay_batch_size + 1))

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
                    if self.args.use_HLOP:
                        data = data.unsqueeze(1)
                        data = data.repeat(1, self.timesteps, 1, 1, 1)
                        out_, out = self.local_model(data, replay_task, projection=False, update_hlop=False)
                    else:
                        out_, out = self.local_model(data)
                    loss = self.loss(out, target)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * target.numel()

                    prec1, prec5 = accuracy(out.data, target.data, topk=(1, 5))
                    losses.update(loss, data.size(0))
                    top1.update(prec1.item(), data.size(0))
                    top5.update(prec5.item(), data.size(0))
                    train_num += target.numel()
                    train_acc += (out.argmax(1) == target).float().sum().item()

                    batch_time.update(time.time() - end)
                    end = time.time()

                    bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx,
                            size=((num_trainset - 1) // self.replay_batch_size + 1) * self.replay_local_epochs,
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
