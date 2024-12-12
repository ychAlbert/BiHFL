#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : SCAFFOLD算法的服务器类
import copy
import os
import time

import numpy as np
from tensorboardX import SummaryWriter

from ..clients.clientscaffold import clientSCAFFOLD
from ..servers.serverbase import Server

__all__ = ['SCAFFOLD']


class SCAFFOLD(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model)
        self.set_clients(clientSCAFFOLD, self.trainsets, taskcla, model)
        self.global_lr = args.SCAFFOLD_eta              # 全局学习率
        self.global_controls = []                       # 全局控制变量
        

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

    def aggregate_parameters(self, delta_model, delta_controls):
        """
        SCAFFOLD聚合参数, 对应论文算法1的16-17行
        @return:
        """
        global_model = self.global_model.state_dict()
        global_controls = self.global_controls.state_dict()
        
        # 计算聚合后的全局模型和控制参数
        for idx in self.received_info['client_ids']:
            delta_yi, delta_c = self.clients[idx].delta_yc()
            for param in global_model:
                delta_model[param] += delta_yi[param]
                delta_controls[param] += delta_c[param]

        for param in global_model:
            delta_model[param] == delta_yi[param] / self.n_client
            delta_controls[param] == delta_controls[param] / self.n_client
            
            global_model[param] += self.global_lr * delta_model[param]
            global_controls[param] += delta_controls[param]
        
        self.global_model.load_state_dict(global_model)
        self.global_controls.load_state_dict(global_controls)

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

            self.global_controls = copy.deepcopy(self.global_model)         # 全局控制参数
            delta_model = self.global_model.state_dict()                    # 记录全局模型的变化
            delta_controls = self.global_model.state_dict()                 # 记录全局控制参数的变化

            for client in self.clients:
                if self.args.use_replay:
                    client.set_replay_data(task_id, n_task_class)           # 如果replay，设置replay的数据
                client.set_optimizer(task_id, False)                        # 客户端设置优化器
                client.set_learning_rate_scheduler(False)                   # 客户端设置学习率

            for global_round in range(1, self.global_rounds + 1):
                for param in delta_model:
                    delta_model[param]  = 0.0
                    delta_controls[param] = 0.0

                self.select_clients(task_id)                                            # 挑选合适客户端
                self.send_models()                                                      # 服务器向选中的客户端发放全局模型
                for client in self.clients:                                             # 选中的客户端进行训练
                    client.train(task_id)
                self.receive_models()                                                   # 服务器接收训练后的客户端模型
                self.aggregate_parameters(delta_model, delta_controls)                  # 服务器聚合全局模型

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

                for replay_global_round in range(1, self.replay_global_rounds + 1):
                    for param in delta_model:
                        delta_model[param]  = 0.0
                        delta_controls[param] = 0.0

                    self.select_clients(task_id)
                    self.send_models()
                    for client in self.clients:
                        client.replay(task_learned)
                    self.receive_models()
                    self.aggregate_parameters(delta_model, delta_controls)

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

