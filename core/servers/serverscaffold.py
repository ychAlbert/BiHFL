#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : SCAFFOLD算法的服务器类
import copy
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from core.clients.clientscaffold import clientSCAFFOLD
from core.servers.serverbase import Server

__all__ = ['SCAFFOLD']


class SCAFFOLD(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model)
        self.set_clients(clientSCAFFOLD, self.trainsets, taskcla, model)
        # 全局学习率
        self.eta = args.SCAFFOLD_eta
        # 全局控制变量
        self.global_c = None
        self.layer_param_num_map = None

    def send_models(self):
        """
        向客户端发送全局模型
        @return:
        """
        # 断言服务器的客户端数不为零
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model, self.global_c)

    def aggregate_parameters(self):
        """
        SCAFFOLD聚合参数, 对应论文算法1的16-17行
        @return:
        """
        global_model = copy.deepcopy(self.global_model)
        global_c = copy.deepcopy(self.global_c)
        for idx in self.received_info['client_ids']:
            dy, dc = self.clients[idx].delta_yc()
            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() / self.n_client * self.eta
            for server_param, client_param in zip(global_c, dc):
                server_param.data += client_param.data.clone() / self.n_client
        self.global_model = global_model
        self.global_c = global_c

    def execute(self):
        # 根据实验名调整重放的决定（如果是bptt/ottt实验，那么一定不重放，其余则根据参数replay的值决定是否重放）>>>>>>>>>>>>>>>>>>>>>>>>>>
        bptt = True if self.args.experiment_name.endswith('bptt') else False
        ottt = True if self.args.experiment_name.endswith('ottt') else False
        if bptt or ottt:
            self.args.use_replay = False

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
                    client.set_replay_data(task_id, n_task_class)  # 如果replay，设置replay的数据
                client.set_optimizer(task_id, False)  # 客户端设置优化器
                client.set_learning_rate_scheduler(False)  # 客户端设置学习率

            for global_round in range(1, self.global_rounds + 1):
                self.update_global_control()
                self.select_clients(task_id)  # 挑选合适客户端
                self.send_models()  # 服务器向选中的客户端发放全局模型
                for client in self.clients:  # 选中的客户端进行训练
                    client.train(task_id, bptt, ottt)
                self.receive_models()  # 服务器接收训练后的客户端模型
                self.aggregate_parameters()  # 服务器聚合全局模型

                print(f"\n-------------Task: {task_id}     Round number: {global_round}-------------")
                print("\033[93mEvaluating\033[0m")
                test_loss, test_acc = self.evaluate(task_id, bptt, ottt)
                writer.add_scalar('test_loss', test_loss, global_round)
                writer.add_scalar('test_acc', test_acc, global_round)

            jj = 0
            for ii in np.array(task_learned)[0:task_count + 1]:
                _, acc_matrix[task_count, jj] = self.evaluate(ii, bptt, ottt)
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
                    _, acc_matrix[task_count, jj] = self.evaluate(ii, bptt, ottt)
                    jj += 1
                print('Accuracies =')
                for i_a in range(task_count + 1):
                    print('\t', end='')
                    for j_a in range(acc_matrix.shape[1]):
                        print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
                    print()

            task_count += 1

    def update_global_control(self):
        # 获取当前本地模型
        current_local_model = copy.deepcopy(self.global_model)

        # 通过有序字典存放模型（层名称：参数）
        layer_param_num_map_new = OrderedDict()
        global_c_new = []
        start_index = 0
        for name, param in current_local_model.named_parameters():
            param_num = len(param.view(-1))
            layer_param_num_map_new[name] = param_num
            global_c_new.append(torch.zeros_like(param))
            start_index += param_num

        if self.global_c is not None:
            layers = self.layer_param_num_map.keys()
            for idx, (name, param) in enumerate(current_local_model.named_parameters()):
                if name in layers:
                    # 如果对应层的参数数量相等，那么就直接替换
                    if layer_param_num_map_new[name] == self.layer_param_num_map[name]:
                        global_c_new[idx] = self.global_c[idx]
                    # 如果对应层的新参数数量大于旧参数数量，那么就进行填充
                    if layer_param_num_map_new[name] > self.layer_param_num_map[name]:
                        param_temp_new, param_temp_old = global_c_new[idx], self.global_c[idx]
                        m, n = param_temp_new.shape[0], param_temp_new.shape[1]
                        a, b = param_temp_old.shape[0], param_temp_old.shape[1]
                        padding_row = m - a
                        padding_col = n - b
                        param_temp = F.pad(param_temp_old, (0, padding_col, 0, padding_row))
                        global_c_new[idx] = param_temp

        self.global_c = global_c_new
        self.layer_param_num_map = layer_param_num_map_new
