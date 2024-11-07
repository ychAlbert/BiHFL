#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : FedProx算法的客户端类
import time
import copy

import numpy as np
from progress.bar import Bar
from torch.utils.data import DataLoader

from ..utils import AverageMeter
from ..clients.clientbase import Client
from ..utils import accuracy, reset_net


class clientProx(Client):
    def __init__(self, args, id, trainset, local_model, taskcla, **kwargs):
        super().__init__(args, id, trainset, local_model, taskcla, **kwargs)
        self.mu = args.FedProx_mu
        self.global_model = None

    def set_parameters(self, model):
        """
        从服务器接收到全局的模型更新本地的模型
        @param model: 全局的模型
        @return:
        """
        self.global_model = copy.deepcopy(self.local_model)
        for p, gp, lp in zip(model.parameters(), self.global_model.parameters(), self.local_model.parameters()):
            gp.data = p.data.clone()
            lp.data = p.data.clone()

    def train(self, task_id):
        # --------------------------------------------------------------------------------------------------------------
        # 获取实验相关参数（是否是HLOP_SNN相关实验，如果是的话，是否是bptt/ottt相关设置）
        # --------------------------------------------------------------------------------------------------------------
        bptt, ottt = False, False
        if self.args.use_hlop:
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
        trainset = self.trainset[f'task {task_id}']
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        num_trainset = len(trainset)

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

        if task_id != 0 and self.args.use_hlop:
            self.local_model.fix_bn()

        train_num = 0
        train_acc = 0
        train_loss = 0
        batch_idx = 0

        samples_index = np.arange(num_trainset)
        np.random.shuffle(samples_index)

        start_time = time.time()
        for local_epoch in range(1, self.local_epochs + 1):
            for data, label in trainloader:
                data = data.float().to(self.device)
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
                        # 根据FedProx论文计算loss
                        proximal_term = 0.0
                        for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
                            proximal_term += (lp - gp).norm(2)
                        loss = self.loss(out_fr, label) / self.timesteps + (self.mu / 2) * proximal_term

                        loss.backward()
                        total_loss += loss.detach()
                        if self.args.online_update:
                            self.optimizer.step()
                    if not self.args.online_update:
                        self.optimizer.step()
                    train_loss += total_loss.item() * label.numel()
                    out = total_fr
                elif bptt:
                    self.optimizer.zero_grad()

                    flag = self.args.use_hlop and (local_epoch > self.args.hlop_start_epochs)
                    if task_id == 0:
                        out_, out = self.local_model(data, task_id, projection=False, update_hlop=flag)
                    else:
                        out_, out = self.local_model(data, task_id, projection=self.args.use_hlop, proj_id_list=[0], update_hlop=flag, fix_subspace_id_list=[0])

                    # 根据FedProx论文计算loss
                    proximal_term = 0.0
                    for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
                        proximal_term += (lp - gp).norm(2)
                    loss = self.loss(out, label) + (self.mu / 2) * proximal_term
                    # loss反向传播
                    loss.backward()
                    # 参数更新
                    self.optimizer.step()
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

                    # 根据FedProx论文计算loss
                    proximal_term = 0.0
                    for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
                        proximal_term += (lp - gp).norm(2)
                    loss = self.loss(out, label) + (self.mu / 2) * proximal_term
                    # loss反向传播
                    loss.backward()
                    # 参数更新
                    self.optimizer.step()
                    train_loss += loss.item() * label.numel()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                losses.update(loss, data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))

                train_num += label.numel()
                train_acc += (out.argmax(1) == label).float().sum().item()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
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

        train_loss /= train_num
        train_acc /= train_num
        self.learning_rate_scheduler.step()
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['num_rounds'] += 1

    def replay(self, tasks_learned):
        self.local_model.train()
        self.local_model.fix_bn()

        for replay_task in tasks_learned:
            replay_trainset = self.replay_trainset[f'task {replay_task}']
            replay_trainloader = DataLoader(replay_trainset, batch_size=self.replay_batch_size, shuffle=True, drop_last=False)
            num_replay_trainset = len(replay_trainset)

            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            end = time.time()
            bar = Bar('Client {:^3d} Replaying Task {:^2d}'.format(self.id, replay_task), max=((num_replay_trainset - 1) // self.replay_batch_size + 1))

            train_num = 0
            train_acc = 0
            train_loss = 0
            batch_idx = 0

            for epoch in range(1, self.replay_local_epochs + 1):
                for data, label in replay_trainloader:
                    data = data.float().to(self.device)
                    label = label.to(self.device)

                    self.optimizer.zero_grad()

                    data = data.unsqueeze(1)
                    data = data.repeat(1, self.timesteps, 1, 1, 1)
                    out_, out = self.local_model(data, replay_task, projection=False, update_hlop=False)

                    # 根据FedProx论文计算loss
                    proximal_term = 0.0
                    for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
                        proximal_term += (lp - gp).norm(2)
                    loss = self.loss(out, label) + (self.mu / 2) * proximal_term
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * label.numel()

                    prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                    losses.update(loss, data.size(0))
                    top1.update(prec1.item(), data.size(0))
                    top5.update(prec5.item(), data.size(0))
                    train_num += label.numel()
                    train_acc += (out.argmax(1) == label).float().sum().item()

                    batch_time.update(time.time() - end)
                    end = time.time()

                    bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=batch_idx,
                            size=((num_replay_trainset - 1) // self.replay_batch_size + 1) * self.replay_local_epochs,
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
