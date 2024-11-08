#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : 服务器的基础类
import copy
import os
import random
import time

import numpy as np
import torch
from progress.bar import Bar
from torch import nn
from torch.utils.data import DataLoader

from FLcore.utils import AverageMeter, accuracy, reset_net, GeneralDataset, split_trainset_by_dirichlet

__all__ = ['Server']


class Server(object):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model):
        self.args = args
        self.device = args.device
        self.taskcla = taskcla

        self.xtrain = xtrain
        self.ytrain = ytrain
        self.client_trainsets = split_trainset_by_dirichlet(args, xtrain, ytrain, taskcla)

        self.xtest = xtest
        self.ytest = ytest
        self.testset = {f'task {item[0]}': GeneralDataset(data=xtest[item[0]], labels=ytest[item[0]], n_class=item[1])
                        for item in self.taskcla}

        self.global_model = copy.deepcopy(model)
        self.global_model.to(self.device)

        self.loss = nn.CrossEntropyLoss()
        self.batch_size = args.batch_size
        self.global_rounds = args.global_rounds
        self.replay_global_rounds = args.replay_global_rounds
        self.timesteps = args.timesteps

        self.n_client = args.n_client
        self.clients = []
        self.selected_clients = []

        self.received_info = {}

        self.root_path = os.path.join(args.root_path, 'Server')
        self.logs_path = os.path.join(self.root_path, 'logs')
        self.models_path = os.path.join(self.root_path, 'models')

    # ------------------------------------------------------------------------------------------------------------------
    # 设置相关客户端操作
    # ------------------------------------------------------------------------------------------------------------------
    # 生成现有的客户端
    def set_clients(self, clientObj, trainsets, model, taskcla):
        for i in range(self.n_client):
            client = clientObj(args=self.args, id=i, trainset=trainsets[i], local_model=copy.deepcopy(model),
                               taskcla=taskcla)
            self.clients.append(client)

    # ------------------------------------------------------------------------------------------------------------------
    # 联邦主要操作
    # ------------------------------------------------------------------------------------------------------------------

    # 挑选客户端
    def select_clients(self, task_id):
        # 除上述要求外，可挑选的客户端还需要其数据可进行task_id任务
        selective_clients = [client for client in self.clients if task_id in client.local_tasks]
        self.selected_clients = list(np.random.choice(selective_clients, self.n_client, replace=False))

    # 向客户端发送全局模型
    def send_models(self):
        """
        向客户端发送全局模型
        @return:
        """
        # 断言服务器的客户端数不为零
        assert (len(self.selected_clients) > 0)
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            client.send_time_cost['num_rounds'] += 1

    # 从客户端接收训练后的本地模型
    def receive_models(self):
        """
        从选中训练的客户端接收其本地模型
        @return:
        """
        # 断言被选中的客户端数不为零
        assert (len(self.selected_clients) > 0)
        # 计算选中的客户端中的活跃客户端数量
        # 随机采样
        activate_clients = random.sample(self.selected_clients, self.n_client)

        self.received_info = {
            'client_ids': [],
            'client_weights': [],
            'client_models': []
        }

        for client in activate_clients:
            self.received_info['client_ids'].append(client.id)
            self.received_info['client_weights'].append(client.train_samples)
            self.received_info['client_models'].append(client.local_model)

        total_client_train_samples = sum(self.received_info['client_weights'])
        for idx, train_samples in enumerate(self.received_info['client_weights']):
            self.received_info['client_weights'][idx] = train_samples / total_client_train_samples

    def evaluate(self, task_id: int):
        # --------------------------------------------------------------------------------------------------------------
        # 获取实验相关参数（是否是HLOP_SNN相关实验，如果是的话，是否是bptt/ottt相关设置）
        # --------------------------------------------------------------------------------------------------------------
        bptt, ottt = False, False
        if self.args.experiment_name.endswith('bptt'):
            bptt = True
        elif self.args.experiment_name.endswith('ottt'):
            ottt = True

        # 全局模型开启评估模式
        self.global_model.eval()
        testset = self.testset[f'task {task_id}']
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        n_testset = len(testset)

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Server Testing', max=((n_testset - 1) // self.batch_size + 1))

        n_testdata = 0
        test_acc = 0
        test_loss = 0
        batch_idx = 0

        with torch.no_grad():
            for data, label in testloader:
                data = data.to(self.device)
                label = label.to(self.device)

                if bptt:
                    out_, out = self.global_model(data, task_id, projection=False, update_hlop=False)
                    loss = self.loss(out, label)
                    reset_net(self.global_model)
                elif ottt:
                    loss = 0.
                    for t in range(self.timesteps):
                        if t == 0:
                            out_fr = self.global_model(data, task_id, projection=False, update_hlop=False,
                                                       init=True)
                            total_fr = out_fr.clone().detach()
                        else:
                            out_fr = self.global_model(data, task_id, projection=False, update_hlop=False)
                            total_fr += out_fr.clone().detach()
                        loss += self.loss(out_fr, label).detach() / self.timesteps
                    out_, out = total_fr
                else:
                    data = data.unsqueeze(1)
                    data = data.repeat(1, self.timesteps, 1, 1, 1)
                    out_, out = self.global_model(data, task_id, projection=False, update_hlop=False)
                    loss = self.loss(out, label)

                n_testdata += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out.argmax(1) == label).float().sum().item()

                prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                losses.update(loss, data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))

                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx,
                    size=((n_testset - 1) // self.batch_size + 1),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
                bar.next()
        bar.finish()

        test_acc /= n_testdata
        test_loss /= n_testdata

        print('Test Accuracy: {:.4f}    Test Loss: {:.4f}'.format(test_acc, test_loss))
        return test_loss, test_acc

    # ------------------------------------------------------------------------------------------------------------------
    # 数据保存、加载操作
    # ------------------------------------------------------------------------------------------------------------------
    def save_global_model(self, model_name):
        """
        保存全局模型
        @param model_name: 模型名称（不需要绝对/相对路径）
        @return:
        """
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        torch.save(self.global_model, os.path.join(self.models_path, f'{model_name}.pt'))

    def load_global_model(self, model_name):
        """
        加载全局模型
        @param model_name: 模型名称（不需要绝对/相对路径）
        @return:
        """
        model_abs_path = os.path.join(self.models_path, model_name)
        assert os.path.exists(model_abs_path)
        self.global_model = torch.load(model_abs_path)
