#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : FedAvg算法的服务器类
import copy
import os
import time

import numpy as np
from tensorboardX import SummaryWriter

from ..clients.clientavg import clientAVG
from ..servers.serverbase import Server

__all__ = ['FedAvg']


class FedAvg(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model)
        self.set_clients(clientAVG, self.client_trainsets, model, taskcla)
        self.time_cost = []

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
        # pmnist/pmnist_bptt/pmnist_ottt 实验
        if self.args.experiment_name.startswith('pmnist'):
            hlop_out_num = [80, 200, 100]
            hlop_out_num_inc = [70, 70, 70]
        # cifar 实验
        elif self.args.experiment_name == 'cifar':
            hlop_out_num = [6, 100, 200]
            hlop_out_num_inc = [2, 20, 40]
        # miniimagenet 实验
        elif self.args.experiment_name == 'miniimagenet':
            hlop_out_num = [24, [90, 90], [90, 90], [90, 180, 10], [180, 180], [180, 360, 20],
                            [360, 360], [360, 720, 40], [720, 720]]
            hlop_out_num_inc = [2, [6, 6], [6, 6], [6, 12, 1], [12, 12], [12, 24, 2], [24, 24], [24, 48, 4], [48, 48]]
            hlop_out_num_inc1 = [0, [2, 2], [2, 2], [2, 4, 0], [4, 4], [4, 8, 0], [8, 8], [8, 16, 0], [16, 16]]
        # fivedataset/fivedataset_domain 实验
        elif self.args.experiment_name.startswith('fivedataset'):
            hlop_out_num = [6, [40, 40], [40, 40], [40, 100, 6], [100, 100], [100, 200, 8],
                            [200, 200], [200, 200, 16], [200, 200]]
            hlop_out_num_inc = [6, [40, 40], [40, 40], [40, 100, 6], [100, 100], [100, 200, 8],
                                [200, 200], [200, 200, 16], [200, 200]]

        task_learned = []
        task_count = 0

        tasks = [task_id for task_id, n_task_class in self.taskcla]
        n_task = len(tasks)

        acc_matrix = np.zeros((n_task, n_task))

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
            # cifar 实验
            elif self.args.experiment_name == 'cifar':
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
            # miniimagenet 实验
            elif self.args.experiment_name == 'miniimagenet':
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
            # fivedataset/fivedataset_domain 实验
            elif self.args.experiment_name.startswith('fivedataset'):
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

            for client in self.clients:
                if self.args.use_replay:
                    client.set_replay_data(task_id, n_task_class)
                client.set_optimizer(task_id, False)
                client.set_learning_rate_scheduler(False)

            for global_round in range(1, self.global_rounds + 1):
                start_time = time.time()
                # ①挑选合适客户端
                self.select_clients(task_id)
                # ②服务器向选中的客户端发放全局模型
                self.send_models()
                # ③选中的客户端进行训练
                for client in self.clients:
                    client.train(task_id)
                # ④服务器接收训练后的客户端模型
                self.receive_models()
                # ⑤服务器聚合全局模型
                self.aggregate_parameters()

                self.time_cost.append(time.time() - start_time)
                print('-' * 10, 'Task', task_id, 'Time Cost: ', self.time_cost[-1], '-' * 10)

                print(f"\n-------------Round number: {global_round}-------------")
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
        """
        根据本地模型聚合全局模型
        @return:
        """
        # 断言客户端上传的模型数量不为零
        assert (len(self.received_info['client_models']) > 0)
        self.global_model = copy.deepcopy(self.received_info['client_models'][0])
        # 将全局模型的参数值清空
        for param in self.global_model.parameters():
            param.data.zero_()
        # 获取全局模型的参数值
        for weight, model in zip(self.received_info['client_weights'], self.received_info['client_models']):
            for server_param, client_param in zip(self.global_model.parameters(), model.parameters()):
                server_param.data += client_param.data.clone() * weight
