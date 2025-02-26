import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cifar100 import get
import copy

# 定义基础CNN网络
class BaseCNN(nn.Module):
    def __init__(self, input_shape, n_outputs):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, n_outputs)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# FedWeIT客户端模型
class FedWeITModel(nn.Module):
    def __init__(self, input_shape, n_outputs, n_tasks=10):
        super(FedWeITModel, self).__init__()
        self.input_shape = input_shape
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks
        
        # 基础特征提取网络
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 特征提取后的维度
        feat_dim = 128 * 4 * 4
        
        # 任务特定的知识库参数
        self.kb_params = nn.ParameterList([
            nn.Parameter(torch.zeros(feat_dim, 256)) for _ in range(n_tasks)
        ])
        
        # 知识库选择器（权重）
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.zeros(n_tasks, feat_dim)) for _ in range(n_tasks)
        ])
        
        # 任务特定的适应层
        self.adapters = nn.ModuleList([
            nn.Linear(256, n_outputs) for _ in range(n_tasks)
        ])
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        
        # 初始化知识库参数
        for kb in self.kb_params:
            nn.init.normal_(kb, 0, 0.01)
            
        # 初始化alpha参数
        for a in self.alpha:
            nn.init.normal_(a, 0, 0.01)
    
    def extract_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # flatten
        return x
    
    def forward(self, x, task_id):
        # 特征提取
        features = self.extract_features(x)
        
        # 基础知识重用
        task_knowledge = 0
        for t in range(self.n_tasks):
            # 只使用当前任务及以前任务的知识
            if t <= task_id:
                weighted_kb = torch.mm(features, self.kb_params[t] * self.alpha[task_id][t].view(-1, 1))
                task_knowledge += weighted_kb
        
        # 任务特定适应
        output = self.adapters[task_id](task_knowledge)
        return output, features, task_knowledge

# FedWeIT客户端类
class FedWeITClient:
    def __init__(self, model, device, task_id, learning_rate=0.001, kb_learning_rate=0.0005):
        self.model = model
        self.device = device
        self.task_id = task_id
        self.optimizer = optim.Adam([
            {'params': self.model.conv1.parameters()},
            {'params': self.model.conv2.parameters()},
            {'params': self.model.conv3.parameters()},
            {'params': self.model.adapters[task_id].parameters()},
        ], lr=learning_rate)
        
        # 知识库参数的优化器
        self.kb_optimizer = optim.Adam([
            {'params': self.model.kb_params[task_id]},
            {'params': self.model.alpha[task_id]},
        ], lr=kb_learning_rate)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, train_loader, old_model=None, lambda_l1=0.001, lambda_kd=1.0, epochs=5):
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                self.kb_optimizer.zero_grad()
                
                # 前向传播
                outputs, features, task_knowledge = self.model(inputs, self.task_id)
                
                # 计算分类损失
                cls_loss = self.criterion(outputs, targets)
                loss = cls_loss
                
                # L1正则化以促进稀疏性
                l1_reg = 0
                for t in range(self.task_id + 1):
                    l1_reg += torch.norm(self.model.alpha[self.task_id][t], 1)
                loss += lambda_l1 * l1_reg
                
                # 知识蒸馏损失（如果有旧模型）
                if old_model is not None and self.task_id > 0:
                    old_model.eval()
                    with torch.no_grad():
                        _, old_features, _ = old_model(inputs, self.task_id - 1)
                    
                    # 重要性加权特征蒸馏
                    kd_loss = F.mse_loss(features, old_features)
                    loss += lambda_kd * kd_loss
                
                loss.backward()
                self.optimizer.step()
                self.kb_optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            acc = 100. * correct / total
            print(f'Epoch: {epoch+1}, Loss: {running_loss/len(train_loader):.3f}, Acc: {acc:.2f}%')
        
        return self.model
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _, _ = self.model(inputs, self.task_id)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f'Test Accuracy: {acc:.2f}%')
        return acc

# 生成模型用于数据重放
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3, img_size=32):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        
        self.init_size = img_size // 4
        self.fc = nn.Linear(latent_dim, 128 * self.init_size * self.init_size)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, self.init_size, self.init_size)
        x = self.conv_blocks(x)
        return x

# FedWeIT服务器类
class FedWeITServer:
    def __init__(self, num_clients, device, latent_dim=100, channels=3, img_size=32):
        self.num_clients = num_clients
        self.device = device
        self.generator = Generator(latent_dim, channels, img_size).to(device)
        self.latent_dim = latent_dim
        self.global_model = None
        self.optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
    def train_generator(self, global_model, task_id, num_samples=1000, batch_size=64, epochs=50):
        """训练生成模型，使用无数据知识蒸馏"""
        self.generator.train()
        global_model.eval()
        
        for epoch in range(epochs):
            g_loss = 0.0
            
            for i in range(0, num_samples, batch_size):
                batch_size_i = min(batch_size, num_samples - i)
                
                # 生成随机噪声
                z = torch.randn(batch_size_i, self.latent_dim).to(self.device)
                
                # 生成合成数据
                gen_imgs = self.generator(z)
                
                # 使用全局模型对生成的数据进行预测
                with torch.no_grad():
                    logits, _, _ = global_model(gen_imgs, task_id)
                    targets = F.softmax(logits, dim=1)
                
                # 创建伪标签（硬标签）
                _, pseudo_labels = targets.max(1)
                
                # 计算交叉熵损失
                outputs, _, _ = global_model(gen_imgs, task_id)
                ce_loss = F.cross_entropy(outputs, pseudo_labels)
                
                # 多样性损失（鼓励生成不同类别的样本）
                probs = F.softmax(outputs, dim=1)
                mean_probs = probs.mean(0)
                diversity_loss = -torch.sum(torch.log(mean_probs + 1e-12))
                
                # 总损失
                loss = ce_loss + 0.1 * diversity_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                g_loss += loss.item()
            
            print(f'Generator Epoch: {epoch+1}, Loss: {g_loss/(num_samples/batch_size):.3f}')
        
        return self.generator
    
    def aggregate_models(self, client_models):
        """聚合客户端模型参数"""
        # 使用FedAvg方法聚合模型
        if self.global_model is None:
            self.global_model = FedWeITModel(client_models[0].input_shape, client_models[0].n_outputs, n_tasks=self.num_clients).to(self.device)
        else:
            # 更新共享参数（特征提取部分）
            with torch.no_grad():
                # 分别处理三个卷积层
                conv_layers = [(self.global_model.conv1, 'conv1'),
                              (self.global_model.conv2, 'conv2'),
                              (self.global_model.conv3, 'conv3')]
                
                for layer, name in conv_layers:
                    avg_weights = torch.zeros_like(layer.weight)
                    avg_bias = torch.zeros_like(layer.bias)
                    
                    for client_model in client_models:
                        client_layer = getattr(client_model, name)
                        avg_weights += client_layer.weight
                        avg_bias += client_layer.bias
                    
                    avg_weights /= len(client_models)
                    avg_bias /= len(client_models)
                    
                    layer.weight.data = avg_weights
                    layer.bias.data = avg_bias
                
                # 知识库参数保持客户端特定
                for t in range(len(self.global_model.kb_params)):
                    for client_id, client_model in enumerate(client_models):
                        if t <= client_id:  # 只考虑当前及之前的任务
                            self.global_model.kb_params[t].data = client_model.kb_params[t].data
                            self.global_model.alpha[t].data = client_model.alpha[t].data
        
        return self.global_model
# 主函数：运行FedWeIT算法
def run_fedweit(data_dir='./data/', num_clients=10, batch_size=64, epochs=5, device='cuda'):
    """运行FedWeIT算法进行联邦持续学习"""
    
    # 检查设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    data, taskcla, size = get(data_dir)
    input_shape = size
    
    # 初始化服务器
    server = FedWeITServer(num_clients, device, channels=input_shape[0], img_size=input_shape[1])
    
    # 初始化客户端模型
    client_models = []
    for client_id in range(num_clients):
        model = FedWeITModel(input_shape, taskcla[client_id][1], n_tasks=num_clients).to(device)
        client_models.append(model)
    
    # 训练过程
    for task_id in range(num_clients):
        print(f"\n--- Training Task {task_id} ({data[task_id]['name']}) ---")
        
        # 为每个客户端创建数据加载器
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(data[task_id]['train']['x'], data[task_id]['train']['y']),
            batch_size=batch_size, shuffle=True
        )
        
        test_loader = DataLoader(
            torch.utils.data.TensorDataset(data[task_id]['test']['x'], data[task_id]['test']['y']),
            batch_size=batch_size, shuffle=False
        )
        
        # 创建客户端对象
        clients = []
        for client_id in range(task_id + 1):  # 只有当前任务及之前的客户端参与
            clients.append(FedWeITClient(client_models[client_id], device, task_id))
        
        # 客户端本地训练
        for i, client in enumerate(clients):
            print(f"Training client {i} for task {task_id}")
            # 获取上一个任务的模型（如果存在）
            old_model = None
            if task_id > 0:
                old_model = client_models[i].clone() if hasattr(client_models[i], 'clone') else copy.deepcopy(client_models[i])
            
            # 训练客户端模型
            client_models[i] = client.train(train_loader, old_model, epochs=epochs)
            
            # 评估客户端模型
            acc = client.evaluate(test_loader)
            print(f"Client {i}, Task {task_id}, Accuracy: {acc:.2f}%")
        
        # 服务器聚合模型
        global_model = server.aggregate_models([client_models[i] for i in range(task_id + 1)])
        
        # 训练生成模型（从任务1开始）
        if task_id > 0:
            generator = server.train_generator(global_model, task_id)
        
        # 评估所有之前的任务（衡量灾难性遗忘）
        if task_id > 0:
            for prev_task in range(task_id):
                prev_test_loader = DataLoader(
                    torch.utils.data.TensorDataset(data[prev_task]['test']['x'], data[prev_task]['test']['y']),
                    batch_size=batch_size, shuffle=False
                )
                
                for i, client in enumerate(clients):
                    if i <= prev_task:  # 只评估曾经学习过该任务的客户端
                        client = FedWeITClient(client_models[i], device, prev_task)
                        acc = client.evaluate(prev_test_loader)
                        print(f"Client {i}, Previous Task {prev_task}, Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    run_fedweit()