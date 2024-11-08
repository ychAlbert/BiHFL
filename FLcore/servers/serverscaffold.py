#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : SCAFFOLD算法的服务器类
import copy
import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from ..clients.clientscaffold import clientSCAFFOLD
from ..servers.serverbase import Server

__all__ = ['SCAFFOLD']


class SCAFFOLD(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model)
        self.set_clients(clientSCAFFOLD, self.client_trainsets, model, taskcla)
        # 全局控制变量
        self.global_controls = []
        self.eta = args.SCAFFOLD_eta
        self.time_cost = []

    def send_models(self):
        """
        向客户端发送全局模型
        @return:
        """
        # 断言服务器的客户端数不为零
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model, self.global_controls)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def aggregate_parameters(self, task_id):
        """
        SCAFFOLD聚合参数
        @return:
        """
        # 全局模型的深拷贝
        global_model = copy.deepcopy(self.global_model)
        # 全局控制参数的深拷贝
        global_controls = copy.deepcopy(self.global_controls)
        # 计算聚合后的全局模型和控制参数
        for idx in self.received_info['client_ids']:
            delta_model, delta_control = self.clients[idx].delta_yc(task_id)
            for global_model_param, local_model_param in zip(global_model.parameters(), delta_model):
                global_model_param.data += local_model_param.data.clone() / self.num_join_clients * self.eta
            for global_control_param, local_control_param in zip(global_controls, delta_control):
                global_control_param.data += local_control_param.data.clone() / self.n_client
        self.global_model = global_model
        self.global_controls = global_controls

    def execute(self):
        # 根据实验名调整重放的决定（如果是bptt/ottt实验，那么一定不重放，其余则根据参数replay的值决定是否重放）
        bptt, ottt = False, False
        if self.args.experiment_name.endswith('bptt'):
            bptt, ottt = True, False
        elif self.args.experiment_name.endswith('ottt'):
            bptt, ottt = False, True

        if bptt or ottt:
            self.args.use_replay = False

        hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1 = [], [], []
        if self.args.experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
            hlop_out_num = [80, 200, 100]
            hlop_out_num_inc = [70, 70, 70]
        elif self.args.experiment_name == 'cifar':  # cifar 实验
            hlop_out_num = [6, 100, 200]
            hlop_out_num_inc = [2, 20, 40]
        elif self.args.experiment_name == 'miniimagenet':  # miniimagenet 实验
            hlop_out_num = [24, [90, 90], [90, 90], [90, 180, 10], [180, 180], [180, 360, 20],
                            [360, 360], [360, 720, 40], [720, 720]]
            hlop_out_num_inc = [2, [6, 6], [6, 6], [6, 12, 1], [12, 12], [12, 24, 2], [24, 24], [24, 48, 4], [48, 48]]
            hlop_out_num_inc1 = [0, [2, 2], [2, 2], [2, 4, 0], [4, 4], [4, 8, 0], [8, 8], [8, 16, 0], [16, 16]]
        elif self.args.experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
            hlop_out_num = [6, [40, 40], [40, 40], [40, 100, 6], [100, 100], [100, 200, 8],
                            [200, 200], [200, 200, 16], [200, 200]]
            hlop_out_num_inc = [6, [40, 40], [40, 40], [40, 100, 6], [100, 100], [100, 200, 8],
                                [200, 200], [200, 200, 16], [200, 200]]

        task_learned = []
        task_count = 0

        tasks = [task_id for task_id, ncla in self.taskcla]
        total_task_num = len(tasks)

        acc_matrix = np.zeros((total_task_num, total_task_num))

        for item in self.taskcla:
            task_id, n_task_class = item[0], item[1]
            task_learned.append(task_id)
            writer = SummaryWriter(os.path.join(self.args.root_path, 'task{task_id}'.format(task_id=task_id)))

            # pmnist/pmnist_bptt/pmnist_ottt 实验
            if self.args.experiment_name.startswith('pmnist'):
                if task_count == 0:
                    self.global_model.add_hlop_subspace(hlop_out_num)
                    self.global_model.to(self.device)
                    for client in self.clients:
                        client.local_model.add_hlop_subspace(hlop_out_num)
                        client.local_model.to(self.device)
                else:
                    if task_count % 3 == 0:
                        hlop_out_num_inc[0] -= 20
                        hlop_out_num_inc[1] -= 20
                        hlop_out_num_inc[2] -= 20
                    self.global_model.add_hlop_subspace(hlop_out_num_inc)
                    for client in self.clients:
                        client.local_model.add_hlop_subspace(hlop_out_num_inc)
            elif self.args.experiment_name == 'cifar':  # cifar 实验
                if task_count == 0:
                    self.global_model.add_hlop_subspace(hlop_out_num)
                    self.global_model.to(self.device)
                    for client in self.clients:
                        client.local_model.add_hlop_subspace(hlop_out_num)
                        client.local_model.to(self.device)
                else:
                    self.global_model.add_classifier(n_task_class)
                    self.global_model.add_hlop_subspace(hlop_out_num_inc)
                    for client in self.clients:
                        client.local_model.add_classifier(n_task_class)
                        client.local_model.add_hlop_subspace(hlop_out_num_inc)
            elif self.args.experiment_name == 'miniimagenet':  # miniimagenet 实验
                if task_count == 0:
                    self.global_model.add_hlop_subspace(hlop_out_num)
                    self.global_model.to(self.device)
                    for client in self.clients:
                        client.local_model.add_hlop_subspace(hlop_out_num)
                        client.local_model.to(self.device)
                else:
                    self.global_model.add_classifier(n_task_class)
                    for client in self.clients:
                        client.local_model.add_classifier(n_task_class)
                    if task_count < 6:
                        self.global_model.add_hlop_subspace(hlop_out_num_inc)
                        for client in self.clients:
                            client.local_model.add_hlop_subspace(hlop_out_num_inc)
                    else:
                        self.global_model.add_hlop_subspace(hlop_out_num_inc1)
                        for client in self.clients:
                            client.local_model.add_hlop_subspace(hlop_out_num_inc1)
            elif self.args.experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
                if task_count == 0:
                    self.global_model.add_hlop_subspace(hlop_out_num)
                    self.global_model.to(self.device)
                    for client in self.clients:
                        client.local_model.add_hlop_subspace(hlop_out_num)
                        client.local_model.to(self.device)
                else:
                    self.global_model.add_classifier(n_task_class)
                    self.global_model.add_hlop_subspace(hlop_out_num_inc)
                    for client in self.clients:
                        client.local_model.add_classifier(n_task_class)
                        client.local_model.add_hlop_subspace(hlop_out_num_inc)

            self.global_controls = [torch.zeros_like(param) for param in self.global_model.parameters()]

            for client in self.clients:
                if self.args.use_replay:
                    client.set_replay_data(task_id, n_task_class)
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
                self.aggregate_parameters(task_id)

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

            self.global_model.to('cpu')
            self.global_model.merge_hlop_subspace()
            self.global_model.to(self.device)
            for client in self.clients:
                client.local_model.to('cpu')
                client.local_model.merge_hlop_subspace()
                client.local_model.to(self.device)

            # 如果重放并且起码参与了一个任务
            if self.args.use_replay and task_count >= 1:
                print('memory replay\n')

                for replay_global_round in range(1, self.replay_global_rounds + 1):
                    self.select_clients(task_id)
                    self.send_models()

                    for client in self.clients:
                        client.replay(task_learned)
                    self.receive_models()
                    self.aggregate_parameters(task_id)

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

