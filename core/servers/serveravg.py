#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : FedAvg算法的服务器类
import os

import numpy as np
from tensorboardX import SummaryWriter

from core.clients.clientavg import clientAVG
from core.servers.serverbase import Server

__all__ = ['FedAvg']


class FedAvg(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model)
        self.set_clients(clientAVG, self.trainsets, taskcla, model)

    def execute(self):
        # 根据实验名调整重放的决定（如果是bptt/ottt实验，那么一定不重放，其余则根据参数replay的值决定是否重放）>>>>>>>>>>>>>>>>>>>>>>>>>>
        bptt = True if self.args.experiment_name.endswith('bptt') else False
        ottt = True if self.args.experiment_name.endswith('ottt') else False
        if bptt or ottt:
            self.args.use_replay = False

        task_learned = []
        task_count = 0

        tasks = [task_id for task_id, n_task_class in self.taskcla]
        n_task = len(tasks)

        acc_matrix = np.zeros((n_task, n_task))

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
                    _, acc_matrix[task_count, jj] = self.evaluate(ii, bptt, ottt)
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

        # 将全局模型的参数值清空
        for param in self.global_model.parameters():
            param.data.zero_()

        # 获取全局模型的参数值
        for weight, model in zip(self.received_info['client_weights'], self.received_info['client_models']):
            for param_global, param_local in zip(self.global_model.parameters(), model.parameters()):
                param_global.data += param_local.data.clone() * weight
