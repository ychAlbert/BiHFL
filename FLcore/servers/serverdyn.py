#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : FedDyn算法的服务器类
import copy
import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from ..clients.clientdyn import clientDyn
from ..servers.serverbase import Server
from ..utils.prepare_utils import prepare_bptt_ottt, prepare_hlop_out


class FedDyn(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model)
        self.set_slow_clients()
        self.set_clients(clientDyn, self.xtrain, self.ytrain, model)
        self.time_cost = []

        self.alpha = args.FedDyn_alpha
        self.server_state = None

    def execute(self,  use_HLOP: bool):
        bptt, ottt = prepare_bptt_ottt(self.args.experiment_name)
        if bptt or ottt:
            self.args.use_replay = False

        hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1 = prepare_hlop_out(self.args.experiment_name)

        task_learned = []
        task_count = 0

        tasks = [task_id for task_id, ncla in self.taskcla]
        total_task_num = len(tasks)

        acc_matrix = np.zeros((total_task_num, total_task_num))

        for task_id, ncla in self.taskcla:
            task_learned.append(task_id)
            writer = SummaryWriter(os.path.join(self.args.root_path, 'task{task_id}'.format(task_id=task_id)))

            # 如果使用HLOP-SNN方法，那么就需要根据相关参数进行调整
            if use_HLOP:
                self.adjust_for_HLOP_before_train_task(ncla, task_count, hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1)

            self.server_state = copy.deepcopy(self.global_model)
            for param in self.server_state.parameters():
                param.data = torch.zeros_like(param.data)

            for client in self.clients:
                if self.args.use_replay:
                    client.set_replay_data(task_id, ncla)
                client.set_optimizer(task_id, False)
                client.set_learning_rate_scheduler(False)

            # 对于任务task_id，进行联邦训练
            for global_round in range(1, self.global_rounds + 1):
                start_time = time.time()
                # ①挑选合适客户端
                self.select_clients(task_id)
                # ②服务器向选中的客户端发放全局模型
                self.send_models()
                # ③选中的客户端进行训练
                for client in self.selected_clients:
                    client.train(task_id)
                # ④服务器接收训练后的客户端模型
                self.receive_models()
                # ⑤服务器聚合全局模型
                self.aggregate_parameters()

                self.time_cost.append(time.time() - start_time)
                print('-' * 25, 'Task', task_id, 'Time Cost', self.time_cost[-1], '-' * 25)

                print(f"\n-------------Round number: {global_round}-------------")
                print("\nEvaluate global model")
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

            if use_HLOP:
                self.adjust_for_HLOP_after_train_task()

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

        # 更新server_state的参数值
        model_delta = copy.deepcopy(self.received_info['client_models'][0])
        for param in model_delta.parameters():
            param.data = torch.zeros_like(param.data)
        for client_model in self.received_info['client_models']:
            for server_param, client_param, delta_param in zip(self.global_model.parameters(),
                                                               client_model.parameters(), model_delta.parameters()):
                delta_param.data += (client_param - server_param) / self.num_clients
        for state_param, delta_param in zip(self.server_state.parameters(), model_delta.parameters()):
            state_param.data -= self.alpha * delta_param

        # 根据server_state更新global_model的参数值
        self.global_model = copy.deepcopy(self.received_info['client_models'][0])
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
        for client_model in self.received_info['client_models']:
            for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() / self.num_join_clients
        for server_param, state_param in zip(self.global_model.parameters(), self.server_state.parameters()):
            server_param.data -= (1 / self.alpha) * state_param
