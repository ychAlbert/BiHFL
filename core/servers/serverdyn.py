#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : FedDyn算法的服务器类
import copy
import os
from collections import OrderedDict

import numpy as np
import torch
from tensorboardX import SummaryWriter

from core.clients.clientdyn import clientDyn
from core.servers.serverbase import Server


class FedDyn(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model)
        self.set_clients(clientDyn, self.trainsets, taskcla, model)
        self.alpha = args.FedDyn_alpha

        self.layers = []
        self.state = OrderedDict()
        for name, param in copy.deepcopy(self.global_model).named_parameters():
            self.layers.append(name)
            self.state[name] = torch.zeros_like(param.data)

    def execute(self):
        self.prepare_hlop_variable()

        task_learned = []
        task_count = 0

        tasks = [task_id for task_id, ncla in self.taskcla]
        total_task_num = len(tasks)

        acc_matrix = np.zeros((total_task_num, total_task_num))

        for item in self.taskcla:
            task_id, n_task_class = item[0], item[1]
            task_learned.append(task_id)
            writer = SummaryWriter(os.path.join(self.args.root_path, 'task{task_id}'.format(task_id=task_id)))

            self.add_subspace_and_classifier(n_task_class, task_count)

            for client in self.clients:
                if self.args.use_replay:
                    client.set_replay_data(task_id, n_task_class)
                client.set_optimizer(task_id, False)
                client.set_learning_rate_scheduler(False)

            for global_round in range(1, self.global_rounds + 1):
                self.select_clients(task_id)  # 挑选合适客户端
                self.send_models()  # 服务器向选中的客户端发放全局模型
                for client in self.selected_clients:  # 选中的客户端进行训练
                    client.update_grad()
                    client.train(task_id)
                self.receive_models()  # 服务器接收训练后的客户端模型
                self.update_state()
                self.aggregate_parameters()  # 服务器聚合全局模型

                print(f"\n-------------Task: {task_id}     Round number: {global_round}-------------")
                print("\033[93mEvaluating\033[0m")
                test_loss, test_acc = self.evaluate(task_id)
                writer.add_scalar('test_loss', test_loss, global_round)
                writer.add_scalar('test_acc', test_acc, global_round)

            jj = 0
            for ii in np.array(task_learned)[0:task_count + 1]:
                _, acc_matrix[task_count, jj] = self.evaluate(ii)
                jj += 1
            print('Accuracies =')
            for i_a in range(task_count + 1):
                print('\t', end='')
                for j_a in range(acc_matrix.shape[1]):
                    print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
                print()

            self.merge_subspace()

            # 如果重放并且起码参与了一个任务
            if self.args.use_replay and task_count >= 1:
                print('memory replay\n')
                for client in self.clients:
                    client.set_optimizer(task_id, True)
                    client.set_learning_rate_scheduler(True)

                for replay_global_round in range(1, self.replay_global_rounds + 1):
                    self.select_clients(task_id)
                    self.send_models()

                    for client in self.clients:
                        client.replay(task_learned)
                    self.receive_models()
                    self.aggregate_parameters()

                # 保存准确率
                jj = 0
                for ii in np.array(task_learned)[0:task_count + 1]:
                    _, acc_matrix[task_count, jj] = self.evaluate(ii)
                    jj += 1
                print('Accuracies =')
                for i_a in range(task_count + 1):
                    print('\t', end='')
                    for j_a in range(acc_matrix.shape[1]):
                        print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
                    print()

            task_count += 1

    def aggregate_parameters(self):
        assert (len(self.received_info['client_models']) > 0)

        self.global_model = copy.deepcopy(self.received_info['client_models'][0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.received_info['client_models']:
            for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() / self.n_client

        for name, server_param in self.global_model.named_parameters():
            self.state[name].data -= self.alpha * server_param
            server_param.data -= (1 / self.alpha) * self.state[name]

    def update_state(self):
        assert (len(self.received_info['client_models']) > 0)

        # 获取当前本地模型
        # current_global_model = copy.deepcopy(self.global_model)
        current_global_model = copy.deepcopy(self.received_info['client_models'][0])

        layers_new = []
        state_new = OrderedDict()
        for name, param in copy.deepcopy(self.global_model).named_parameters():
            layers_new.append(name)
            state_new[name] = torch.zeros_like(param.data)

        if self.state is not None:
            for name, param in current_global_model.named_parameters():
                if name in self.layers:
                    len_old = self.state[name].detach().clone().view(-1)
                    len_new = state_new[name].detach().clone().view(-1)
                    if len(len_old) == len(len_new):
                        state_new[name] = self.state[name]

        self.layers = layers_new
        self.state = state_new

        model_delta = copy.deepcopy(self.received_info['client_models'][0])
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)

        for client_model in self.received_info['client_models']:
            for server_param, client_param, delta_param in zip(self.global_model.parameters(),
                                                               client_model.parameters(),
                                                               model_delta.parameters()):
                delta_param.data += (client_param - server_param) / self.n_client

        for name, delta_param in model_delta.named_parameters():
            self.state[name].data -= self.alpha * delta_param
