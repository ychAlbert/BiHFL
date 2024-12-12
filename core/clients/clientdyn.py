#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : FedDyn算法的客户端类
import copy
import time
from collections import OrderedDict

import torch
from progress.bar import Bar
from torch.utils.data import DataLoader

from core.clients.clientbase import Client
from core.utils import AverageMeter, accuracy, reset_net


class clientDyn(Client):
    def __init__(self, args, id, trainset, taskcla, model):
        super().__init__(args, id, trainset, taskcla, model)
        self.alpha = args.FedDyn_alpha

        self.global_model_vector = None
        self.layer_index_map = None
        self.grad = None

    def set_parameters(self, model):
        """
        设置参数
        @param model: 全局模型
        @return:
        """
        for old_param, new_param in zip(self.local_model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
        self.global_model_vector = torch.cat([param.view(-1) for param in model.parameters()], dim=0)

    def train(self, task_id):
        bptt = True if self.args.experiment_name.endswith('bptt') else False  # 是否是bptt实验
        ottt = True if self.args.experiment_name.endswith('ottt') else False  # 是否是ottt实验

        # 数据集相关内容 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        trainset = self.trainset[f'task {task_id}']
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        n_trainset = len(trainset)

        # 进度条相关指标及设置 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Client {:3d} Training'.format(self.id), max=((n_trainset - 1) // self.batch_size + 1))

        # 训练主体部分 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        self.local_model.train()

        if task_id != 0:
            self.local_model.fix_bn()

        n_traindata = 0
        train_acc = 0
        train_loss = 0
        batch_idx = 0

        start_time = time.time()

        # 训练之前先更新一次old_grad，防止网络结构
        # self.update_grad()

        for local_epoch in range(1, self.local_epochs + 1):
            for data, label in trainloader:
                data, label = data.to(self.device), label.to(self.device)

                # OTTT设置 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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

                        if self.global_model_vector is not None:
                            v1 = torch.cat([p.view(-1) for p in self.local_model.parameters()], dim=0)
                            loss += self.alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                            loss -= torch.dot(v1, self.grad)

                        loss.backward()
                        total_loss += loss.detach()
                        if self.args.online_update:
                            self.optimizer.step()
                    if not self.args.online_update:
                        self.optimizer.step()
                    train_loss += total_loss.item() * label.numel()
                    out = total_fr

                # BPTT设置 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                elif bptt:
                    self.optimizer.zero_grad()

                    flag = self.args.use_hlop and (local_epoch > self.args.hlop_start_epochs)
                    if task_id == 0:
                        out_, out = self.local_model(data, task_id, projection=False, update_hlop=flag)
                    else:
                        out_, out = self.local_model(data, task_id, projection=self.args.use_hlop, proj_id_list=[0],
                                                     update_hlop=flag, fix_subspace_id_list=[0])

                    loss = self.loss(out, label)

                    if self.global_model_vector is not None:
                        v1 = torch.cat([p.view(-1) for p in self.local_model.parameters()], dim=0)
                        loss += self.alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                        loss -= torch.dot(v1, self.grad)

                    loss.backward()
                    self.optimizer.step()
                    reset_net(self.local_model)
                    train_loss += loss.item() * label.numel()

                # 正常设置 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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

                    if self.global_model_vector is not None:
                        v1 = torch.cat([param.view(-1) for param in self.local_model.parameters()], dim=0)
                        loss += self.alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                        loss -= torch.dot(v1, self.grad)

                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item() * label.numel()

                prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                losses.update(loss, data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))

                n_traindata += label.numel()
                train_acc += (out.argmax(1) == label).float().sum().item()

                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

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

        if self.global_model_vector is not None:
            v1 = torch.cat([p.view(-1) for p in self.local_model.parameters()], dim=0).detach()
            self.grad = self.grad - self.alpha * (v1 - self.global_model_vector)

        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['num_rounds'] += 1

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

            batch_idx = 0

            for epoch in range(1, self.replay_local_epochs + 1):
                for data, label in replay_trainloader:
                    data, label = data.to(self.device), label.to(self.device)

                    data = data.unsqueeze(1)
                    data = data.repeat(1, self.timesteps, 1, 1, 1)

                    self.optimizer.zero_grad()
                    out_, out = self.local_model(data, replay_task, projection=False, update_hlop=False)
                    loss = self.loss(out, label)

                    # if self.global_model_vector is not None:
                    #     v1 = model_parameter_vector(self.local_model)
                    #     loss += self.alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                    #     loss -= torch.dot(v1, self.old_grad)

                    loss.backward()
                    self.optimizer.step()

                    prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                    losses.update(loss, data.size(0))
                    top1.update(prec1.item(), data.size(0))
                    top5.update(prec5.item(), data.size(0))

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

            # if self.global_model_vector is not None:
            #     v1 = model_parameter_vector(self.local_model).detach()
            #     self.old_grad = self.old_grad - self.alpha * (v1 - self.global_model_vector)

    def update_grad(self):
        """
        按照算法更新旧的grad
        @return:
        """

        # 获取当前本地模型
        current_local_model = copy.deepcopy(self.local_model)

        # 通过有序字典存放模型（层名称：参数）
        layer_index_map_new = OrderedDict()
        grad_new = []
        start_index = 0
        for name, param in current_local_model.named_parameters():
            param_temp = param.view(-1)
            layer_index_map_new[name] = [start_index, start_index + param_temp.shape[0]]
            grad_new.append(param_temp)
            start_index += param_temp.shape[0]
        # 计算模型的梯度
        grad_new = torch.zeros_like(torch.cat(grad_new, dim=0))

        # 更新self.grad >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if self.grad is not None:
            layers = layer_index_map_new.keys()
            for name, param in current_local_model.named_parameters():
                if name in layers:
                    index_new, index_old = layer_index_map_new[name], self.layer_index_map[name]
                    # 考虑以下情况，主要考虑hlop模块相关内容
                    differ = (index_new[1] - index_new[0]) - (index_old[1] - index_old[0])
                    if differ > 0:  # 如果某一层的grad_new大于对应的self.grad，那么使用0进行填充
                        grad_new[index_new[0]: index_new[1]] = torch.cat((self.grad[index_old[0]: index_old[1]],
                                                                          torch.zeros(differ).to(self.grad.device)),
                                                                         dim=0)
                    elif differ == 0:  # 如果某一层的grad_new等于对应的self.grad，那么直接替换
                        grad_new[index_new[0]: index_new[1]] = self.grad[index_old[0]: index_old[1]]
                    elif differ < 0:  # 如果某一层的grad_new小于对应的self.grad，那么直接替换self.grad中的一部分
                        grad_new[index_new[0]: index_new[1]] = self.grad[index_old[0]: index_old[1]-differ]

        self.layer_index_map = layer_index_map_new
        self.grad = grad_new.detach().clone()
