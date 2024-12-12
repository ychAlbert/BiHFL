#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : MOON算法的客户端类
import copy
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from progress.bar import Bar

from core.clients.clientbase import Client
from core.utils import AverageMeter
from core.utils import accuracy, reset_net


class clientMOON(Client):
    def __init__(self, args, id, trainset, taskcla, model):
        super().__init__(args, id, trainset, taskcla, model)
        self.tau = args.MOON_tau
        self.mu = args.MOON_mu

    def train(self, task_id):
        bptt = True if self.args.experiment_name.endswith('bptt') else False  # 是否是bptt实验
        ottt = True if self.args.experiment_name.endswith('ottt') else False  # 是否是ottt实验

        # 数据集相关内容 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        trainset = self.trainset[f'task {task_id}']
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        n_trainset = len(trainset)

        # 进度条相关指标及设置 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Client {:^3d} Training'.format(self.id), max=((n_trainset - 1) // self.batch_size + 1))

        # 训练主体部分 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 开启模型训练模式
        self.local_model.train()
        self.old_local_model = copy.deepcopy(self.local_model)

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
                            rep, out_fr = self.local_model(data, task_id, projection=False, update_hlop=flag, init=init)
                            rep_old, out_fr_old = self.old_local_model(data, task_id, projection=False,
                                                                       update_hlop=flag, init=init)
                            rep_global, out_fr_global = self.global_model(data, task_id, projection=False,
                                                                          update_hlop=flag, init=init)
                        else:
                            rep, out_fr = self.local_model(data, task_id, projection=self.args.use_hlop,
                                                           proj_id_list=[0], update_hlop=flag, fix_subspace_id_list=[0],
                                                           init=init)
                            rep_old, out_fr_old = self.old_local_model(data, task_id, projection=self.args.use_hlop,
                                                                       proj_id_list=[0], update_hlop=flag,
                                                                       fix_subspace_id_list=[0], init=init)
                            rep_global, out_fr_global = self.global_model(data, task_id, projection=self.args.use_hlop,
                                                                          proj_id_list=[0], update_hlop=flag,
                                                                          fix_subspace_id_list=[0], init=init)

                        if t == 0:
                            total_fr = out_fr.clone().detach()
                        else:
                            total_fr += out_fr.clone().detach()
                        loss = self.loss(out_fr, label) / self.timesteps
                        loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (
                                torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(
                            F.cosine_similarity(rep, rep_old) / self.tau)))
                        loss += self.mu * torch.mean(loss_con)
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
                        rep, out = self.local_model(data, task_id, projection=False, update_hlop=flag)
                        rep_old, out_fr_old = self.old_local_model(data, task_id, projection=False, update_hlop=flag)
                        rep_global, out_fr_global = self.global_model(data, task_id, projection=False, update_hlop=flag)
                    else:
                        rep, out = self.local_model(data, task_id, projection=self.args.use_hlop, proj_id_list=[0],
                                                    update_hlop=flag, fix_subspace_id_list=[0])
                        rep_old, out_fr_old = self.old_local_model(data, task_id, projection=self.args.use_hlop,
                                                                   proj_id_list=[0], update_hlop=flag,
                                                                   fix_subspace_id_list=[0])
                        rep_global, out_fr_global = self.global_model(data, task_id, projection=self.args.use_hlop,
                                                                      proj_id_list=[0], update_hlop=flag,
                                                                      fix_subspace_id_list=[0])
                    loss = self.loss(out, label)
                    loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (
                            torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(
                        F.cosine_similarity(rep, rep_old) / self.tau)))
                    loss += self.mu * torch.mean(loss_con)
                    loss.backward()
                    self.optimizer.step()
                    reset_net(self.local_model)
                    train_loss += loss.item() * label.numel()
                else:
                    data = data.unsqueeze(1)
                    data = data.repeat(1, self.timesteps, 1, 1, 1)
                    self.optimizer.zero_grad()

                    flag = self.args.use_hlop and (local_epoch > self.args.hlop_start_epochs)
                    if task_id == 0:
                        rep, out = self.local_model(data, task_id, projection=False, update_hlop=flag)
                        rep_old, out_fr_old = self.old_local_model(data, task_id, projection=False, update_hlop=flag)
                        rep_global, out_fr_global = self.global_model(data, task_id, projection=False, update_hlop=flag)
                    else:
                        rep, out = self.local_model(data, task_id, projection=self.args.use_hlop, proj_id_list=[0],
                                                    update_hlop=flag, fix_subspace_id_list=[0])
                        rep_old, out_fr_old = self.old_local_model(data, task_id, projection=self.args.use_hlop,
                                                                   proj_id_list=[0], update_hlop=flag,
                                                                   fix_subspace_id_list=[0])
                        rep_global, out_fr_global = self.global_model(data, task_id, projection=self.args.use_hlop,
                                                                      proj_id_list=[0], update_hlop=flag,
                                                                      fix_subspace_id_list=[0])
                    loss = self.loss(out, label)
                    loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (
                            torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(
                        F.cosine_similarity(rep, rep_old) / self.tau)))
                    loss += self.mu * torch.mean(loss_con)
                    loss.backward()
                    self.optimizer.step()
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
        self.lr_scheduler.step()
        self.old_local_model = copy.deepcopy(self.local_model)
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['num_rounds'] += 1

    def replay(self, tasks_learned):
        self.local_model.train()
        self.local_model.fix_bn()

        for replay_task in tasks_learned:
            replay_trainset = self.replay_trainset[f'task {replay_task}']
            replay_trainloader = DataLoader(replay_trainset, batch_size=self.replay_batch_size,
                                            shuffle=True, drop_last=False)
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

            for epoch in range(1, self.replay_local_epochs + 1):
                for data, label in replay_trainloader:
                    data = data.to(self.device)
                    label = label.to(self.device)

                    self.optimizer.zero_grad()

                    data = data.unsqueeze(1)
                    data = data.repeat(1, self.timesteps, 1, 1, 1)

                    rep, out = self.local_model(data, replay_task, projection=False, update_hlop=False)
                    rep_old, out_fr_old = self.old_local_model(data, replay_task, projection=False, update_hlop=False)
                    rep_global, out_fr_global = self.global_model(data, replay_task, projection=False,
                                                                  update_hlop=False)

                    # 计算loss
                    loss = self.loss(out, label)

                    loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (
                            torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(
                        F.cosine_similarity(rep, rep_old) / self.tau)))
                    loss += self.mu * torch.mean(loss_con)

                    loss.backward()
                    self.optimizer.step()
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

    def set_parameters(self, model):
        """
        根据接收到的模型参数设置相关参数
        :param model: 接收到的模型
        :return:
        """
        # 更新本地模型参数
        for old_param, new_param in zip(self.local_model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
        # 更新本地存放的全局模型
        self.global_model = copy.deepcopy(model)
