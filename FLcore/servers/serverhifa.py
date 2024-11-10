import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import copy
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from ..clients.clientavg import clientAVG
from ..servers.serverbase import Server


class AdaptiveNeuronLayer(nn.Module):
    """
    适应性神经元层，模拟生物神经元的动态特性
    - 整合发火模型(Integrate-and-Fire)
    - 突触可塑性(Synaptic Plasticity)
    - 自适应阈值(Adaptive Threshold)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 突触权重
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
        
        # 静息电位和发放阈值
        self.resting_potential = -70.0  # mV
        self.threshold_baseline = -55.0  # mV
        
        # 自适应阈值参数
        self.threshold = nn.Parameter(torch.ones(output_dim) * self.threshold_baseline)
        self.adaptation_rate = 0.01
        
        # 突触可塑性参数
        self.learning_rate_stdp = 0.001
        self.time_window = 20.0  # ms
        
        # 膜电位衰减参数
        self.decay_rate = 0.9
        
        # 突触整合参数
        self.integration_window = nn.Parameter(torch.ones(output_dim) * 0.1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 计算突触后电位
        post_synaptic_potential = F.linear(x, self.weights)
        
        # 添加静息电位
        membrane_potential = self.resting_potential + post_synaptic_potential
        
        # 应用膜电位衰减
        membrane_potential = self.decay_rate * membrane_potential
        
        # 检查是否达到发放阈值
        spike_mask = (membrane_potential >= self.threshold).float()
        spikes = spike_mask * membrane_potential
        
        # 自适应阈值更新
        self.threshold.data += self.adaptation_rate * (spike_mask - self.threshold)
        
        # 应用突触可塑性
        if self.training:
            self._apply_stdp(x, spikes)
        
        return spikes, membrane_potential
    
    def _apply_stdp(self, pre_synaptic: torch.Tensor, post_synaptic: torch.Tensor):
        """实现简化的STDP(Spike-Timing-Dependent Plasticity)"""
        time_diff = torch.matmul(pre_synaptic.t(), post_synaptic)
        stdp_factor = torch.exp(-torch.abs(time_diff) / self.time_window)
        weight_update = self.learning_rate_stdp * stdp_factor * torch.sign(time_diff)
        self.weights.data += weight_update


class BiologicalSWR(nn.Module):
    """
    基于生物学的Sharp Wave-Ripple (SWR)模型
    - 模拟CA3-CA1网络动态
    - 实现快速序列压缩和重放
    - 动态调节网络活动
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # 动态计算中间层维度
        self.hidden_dim = max(32, min(256, input_dim // 64))
        
        # CA3样式编码层
        self.ca3_encoder = AdaptiveNeuronLayer(input_dim, self.hidden_dim)
        
        # CA1样式解码层
        self.ca1_decoder = AdaptiveNeuronLayer(self.hidden_dim, input_dim)
        
        # 动态调节参数
        self.ripple_frequency = 200.0  # Hz
        self.ripple_duration = 100.0   # ms
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 生成时间窗口
        batch_size = x.size(0) if len(x.shape) > 1 else 1
        time_steps = int(self.ripple_duration * self.ripple_frequency / 1000)
        
        # CA3编码
        ca3_spikes, ca3_potentials = self.ca3_encoder(x)
        
        # 生成ripple振荡
        ripple_phase = torch.linspace(0, 2*np.pi, time_steps)
        ripple_oscillation = torch.sin(ripple_phase).to(x.device)
        
        # 调制CA3输出
        modulated_spikes = ca3_spikes * ripple_oscillation.view(-1, 1)
        
        # CA1解码
        ca1_spikes, _ = self.ca1_decoder(modulated_spikes)
        
        # 整合输出
        output = self.activation(ca1_spikes.mean(dim=0) if len(ca1_spikes.shape) > 1 else ca1_spikes)
        
        return output


class BiologicalBARR(nn.Module):
    """
    基于生物学的Burst Action Potential Events (BARR)模型
    - 模拟CCK+篮状细胞的突发性活动
    - 实现活动依赖性抑制
    - 动态阈值调节
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # 动态计算中间层维度
        self.hidden_dim = max(32, min(256, input_dim // 64))
        
        # 突发检测层
        self.burst_detector = AdaptiveNeuronLayer(input_dim, self.hidden_dim)
        
        # 抑制生成层
        self.inhibition_generator = AdaptiveNeuronLayer(self.hidden_dim, input_dim)
        
        # 突发参数
        self.burst_threshold = nn.Parameter(torch.ones(1) * 0.5)
        self.burst_adaptation_rate = 0.1
        self.inhibition_strength = nn.Parameter(torch.ones(1) * 0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 检测突发活动
        burst_spikes, burst_potentials = self.burst_detector(x)
        
        # 计算突发概率
        burst_probability = torch.sigmoid(burst_potentials - self.burst_threshold)
        
        # 生成抑制信号
        inhibition_spikes, _ = self.inhibition_generator(burst_spikes)
        
        # 调制抑制强度
        modulated_inhibition = inhibition_spikes * self.inhibition_strength * burst_probability
        
        # 更新突发阈值
        if self.training:
            self.burst_threshold.data += self.burst_adaptation_rate * (burst_probability.mean() - 0.5)
        
        return torch.sigmoid(modulated_inhibition)


class HIFA(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model)
        self.set_clients(clientAVG, self.client_trainsets, model, taskcla)
        self.time_cost = []
        
        # 初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_size = sum(p.numel() for p in self.global_model.parameters())
        
        # 初始化生物学启发的模型
        self.swr_model = BiologicalSWR(model_size).to(self.device)
        self.barr_model = BiologicalBARR(model_size).to(self.device)
        
        # 优化器设置
        self.swr_optimizer = torch.optim.Adam(self.swr_model.parameters(), lr=0.001)
        self.barr_optimizer = torch.optim.Adam(self.barr_model.parameters(), lr=0.001)
        
        # 参数设置
        self.beta = args.beta if hasattr(args, 'beta') else 0.3
        self.consolidation_rate = args.consolidation_rate if hasattr(args, 'consolidation_rate') else 0.1
        
    def update_memory_models(self, updates_tensor):
        """更新神经科学模型"""
        # SWR处理
        swr_output = self.swr_model(updates_tensor)
        
        # BARR处理
        barr_output = self.barr_model(updates_tensor)
        
        # 计算损失
        swr_loss = F.mse_loss(swr_output, updates_tensor)
        barr_loss = F.mse_loss(barr_output, updates_tensor)
        
        # 更新SWR模型
        self.swr_optimizer.zero_grad()
        swr_loss.backward()
        self.swr_optimizer.step()
        
        # 更新BARR模型
        self.barr_optimizer.zero_grad()
        barr_loss.backward()
        self.barr_optimizer.step()
        
        return swr_loss.item(), barr_loss.item()

    def aggregate_parameters(self):
        """使用生物学启发的聚合方法"""
        assert len(self.received_info['client_models']) > 0
        
        # 计算模型更新
        updates = []
        for model in self.received_info['client_models']:
            update = self.calculate_model_update(model)
            updates.append(update)
        
        updates_tensor = torch.stack(updates)
        
        # 应用SWR和BARR处理
        swr_signal = self.swr_model(updates_tensor)
        barr_signal = self.barr_model(updates_tensor)
        
        # 计算最终更新
        consolidated_update = swr_signal * (1 - self.beta * barr_signal)
        
        # 应用更新
        global_params = self.flatten_parameters(self.global_model)
        updated_params = global_params + self.consolidation_rate * consolidated_update
        self.unflatten_parameters(updated_params, self.global_model)
        
        # 更新内存模型
        self.update_memory_models(updates_tensor)
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
