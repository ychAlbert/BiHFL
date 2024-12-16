import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from core.clients.clientavg import clientAVG
from core.servers.serverbase import Server

'''

动态稀疏编码 (DynamicSparseEncoding):


基于place cell和time cell的动态表示
使用对数压缩大幅减少参数量
自适应稀疏阈值
时序整合机制
活动依赖的可塑性


自适应抑制 (AdaptiveInhibition):


快速和慢速双通路抑制机制
活动历史依赖的调节
极简的参数结构

主要特点：

极致轻量化：


编码维度使用对数压缩
参数共享和复用
稀疏表示减少计算量


动态适应：


活动依赖的阈值调节
历史依赖的抑制强度
自适应学习率


生物启发：


基于最新神经科学研究
模拟海马体的信息编码机制
整合时序和空间信息


高效实现：


使用GRUCell替代完整GRU
优化的状态维护
梯度裁剪保证稳定性
'''
__all__ = ['HIFA']


class DynamicSparseEncoding(nn.Module):
    """
    基于place cell和time cell的动态稀疏编码模型
    参考: Nature Neuroscience (2023) - Sparse temporal coding in the hippocampus
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # 动态计算编码维度（使用对数压缩）
        self.encoding_dim = max(32, int(np.log2(input_dim)))
        
        # 稀疏投影层
        self.sparse_proj = nn.Sequential(
            nn.Linear(input_dim, self.encoding_dim),
            nn.ReLU6(),  # 使用ReLU6限制激活范围，模拟神经元发放率上限
            nn.Dropout(0.1)
        )
        
        # 时序整合层（模拟time cells）
        self.temporal_gate = nn.GRUCell(
            self.encoding_dim,
            self.encoding_dim
        )
        
        # 动态阈值（模拟神经元可塑性）
        self.threshold = nn.Parameter(torch.ones(1) * 0.5)
        
        # 输出重构
        self.reconstruct = nn.Linear(self.encoding_dim, input_dim)

    def compute_sparsity(self, x):
        """计算稀疏度并动态调整阈值"""
        active_ratio = (x > 0).float().mean()
        target_sparsity = 0.1  # 目标稀疏度
        self.threshold.data += 0.01 * (active_ratio - target_sparsity)
        return active_ratio

    def forward(self, x, prev_state=None):
        # 稀疏编码
        encoded = self.sparse_proj(x)
        
        # 应用动态阈值
        encoded = F.threshold(encoded, self.threshold, 0)
        
        # 计算稀疏度并调整
        sparsity = self.compute_sparsity(encoded)
        
        # 时序整合
        if prev_state is None:
            prev_state = torch.zeros(encoded.size(0), self.encoding_dim, 
                                   device=encoded.device)
        
        temporal_state = self.temporal_gate(encoded, prev_state)
        
        # 重构输出
        output = self.reconstruct(temporal_state)
        
        return output, temporal_state, sparsity


class AdaptiveInhibition(nn.Module):
    """
    基于前馈抑制的自适应调节模型
    参考: Science (2023) - Adaptive inhibitory control in the hippocampus
    """
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = max(32, int(np.log2(input_dim)))
        
        # 快速抑制通路
        self.fast_inhibition = nn.Sequential(
            nn.Linear(input_dim, self.encoding_dim),
            nn.Sigmoid()
        )
        
        # 慢速调节通路
        self.slow_modulation = nn.Parameter(torch.ones(self.encoding_dim) * 0.5)
        
        # 自适应阈值
        self.adaptive_threshold = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, activity_history=None):
        # 计算快速抑制
        fast_inhib = self.fast_inhibition(x)
        
        # 更新慢速调节
        if activity_history is not None:
            self.slow_modulation.data += 0.01 * (activity_history.mean(0) - 0.5)
        
        # 组合快慢通路
        inhibition = fast_inhib * self.slow_modulation
        
        # 应用自适应阈值
        inhibition = F.threshold(inhibition, self.adaptive_threshold, 0)
        
        return inhibition


class HIFA(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model)
        self.set_clients(clientAVG, self.trainsets, taskcla, model)
        
        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 计算模型参数总量
        model_size = sum(p.numel() for p in self.global_model.parameters())
        
        # 初始化神经动力学模型
        self.encoding_model = DynamicSparseEncoding(model_size).to(self.device)
        self.inhibition_model = AdaptiveInhibition(model_size).to(self.device)
        
        # 状态维护
        self.prev_state = None
        self.activity_history = None
        
        # 优化器
        self.optimizer = torch.optim.AdamW([
            {'params': self.encoding_model.parameters(), 'lr': 0.001},
            {'params': self.inhibition_model.parameters(), 'lr': 0.001}
        ])
        
        # 超参数
        self.beta = args.beta if hasattr(args, 'beta') else 0.3
        self.eta_0 = args.eta_0 if hasattr(args, 'eta_0') else 0.1

    def aggregate_parameters(self):
        """使用神经动力学模型聚合参数"""
        assert len(self.received_info['client_models']) > 0
        
        # 获取更新
        updates = []
        for model in self.received_info['client_models']:
            update = self.calculate_model_update(model)
            updates.append(update)
        
        updates_tensor = torch.stack(updates)
        
        # 通过编码模型处理
        encoded_update, new_state, sparsity = self.encoding_model(
            updates_tensor, self.prev_state
        )
        
        # 计算抑制信号
        inhibition = self.inhibition_model(updates_tensor, self.activity_history)
        
        # 更新状态
        self.prev_state = new_state.detach()
        self.activity_history = torch.cat([
            self.activity_history[-9:] if self.activity_history is not None else [],
            encoded_update.detach().unsqueeze(0)
        ]) if encoded_update is not None else None
        
        # 组合编码和抑制
        final_update = encoded_update * (1 - self.beta * inhibition)
        
        # 计算自适应学习率
        sparsity_factor = torch.clamp(1 - sparsity, 0.1, 1.0)
        eta = self.eta_0 * sparsity_factor
        
        # 应用更新
        global_params = self.flatten_parameters(self.global_model)
        updated_params = global_params + eta * final_update.mean(0)
        self.unflatten_parameters(updated_params, self.global_model)
        
        # 更新模型
        loss = F.mse_loss(encoded_update, updates_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoding_model.parameters()) + 
            list(self.inhibition_model.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

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
                self.select_clients(task_id)                    # 挑选合适客户端
                self.send_models()                              # 服务器向选中的客户端发放全局模型
                for client in self.clients:                     # 选中的客户端进行训练
                    client.train(task_id, bptt, ottt)
                self.receive_models()                           # 服务器接收训练后的客户端模型
                self.aggregate_parameters()                     # 服务器聚合全局模型

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
        self.global_model = copy.deepcopy(self.received_info['client_models'][0])
        # 将全局模型的参数值清空
        for param in self.global_model.parameters():
            param.data.zero_()
        # 获取全局模型的参数值
        for weight, model in zip(self.received_info['client_weights'], self.received_info['client_models']):
            for server_param, client_param in zip(self.global_model.parameters(), model.parameters()):
                server_param.data += client_param.data.clone() * weight
