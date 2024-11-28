#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : 程序设置

import argparse
import os
import time

import torch
import yaml

parser = argparse.ArgumentParser()
# HLOP—SNN 设置 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
parser.add_argument('-timesteps', type=int, default=20)
parser.add_argument('-Vth', type=float, default=0.3)
parser.add_argument('-tau', type=float, default=1.0)
parser.add_argument('-delta_t', type=float, default=0.05)
parser.add_argument('-alpha', type=float, default=0.3)
parser.add_argument('-train_Vth', type=int, default=1, )
parser.add_argument('-Vth_bound', type=float, default=0.0005)
parser.add_argument('-rate_stat', type=int, default=0)

parser.add_argument('-not_hlop_with_wfr', action='store_true', help='use spikes for hlop update')
parser.add_argument('-hlop_spiking', action='store_true', help='use hlop with lateral spiking neurons')
parser.add_argument('-hlop_spiking_scale', type=float, default=20.)
parser.add_argument('-hlop_spiking_timesteps', type=float, default=1000.)
parser.add_argument('--hlop_proj_type', type=str, default='input',
                    help='在底层实现中选择投影类型，默认为输入，可以选择权重来加速卷积运算')
parser.add_argument('--feedback_alignment', action='store_true', help='feedback alignment')
parser.add_argument('--sign_symmetric', action='store_true', help='use sign symmetric')
# HLOP—SNN 设置 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# 联邦算法相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
parser.add_argument('--FedDyn_alpha', type=float, default=1.0, help='FedDyn算法的α参数')
parser.add_argument('--FedProx_mu', type=float, default=0.1, help='FedProx算法的μ参数')
parser.add_argument('--MOON_tau', type=float, default=1.0, help='MOON算法的τ参数')
parser.add_argument('--MOON_mu', type=float, default=1.0, help='MOON算法的μ参数')
parser.add_argument('--SCAFFOLD_glr', type=float, default=1.0, help='SCAFFOLD算法中的全局学习率')
# 联邦算法相关参数 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# 训练及重放相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
parser.add_argument('--global_rounds', type=int, default=1, help='全局通信轮次')
parser.add_argument('--local_epochs', type=int, default=1, help='本地训练轮次')
parser.add_argument('--batch_size', type=int, default=64, help='批处理大小')

parser.add_argument('--replay_global_rounds', type=int, default=1, help='重放全局通信轮次')
parser.add_argument('--replay_local_epochs', type=int, default=1, help='重放本地训练轮次')
parser.add_argument('--replay_batch_size', type=int, default=64, help='重放批处理大小')

parser.add_argument('--opt', type=str, default='SGD', help='使用的优化器')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD的冲量')
parser.add_argument('--lr', type=float, default=0.01, help='本地学习率')
parser.add_argument('--replay_lr', type=float, default=0.001, help='本地重放学习率')
parser.add_argument('--continual_lr', type=float, default=0.01, help='持续任务的学习率')

parser.add_argument('--lr_scheduler', type=str, default='CosALR', help='学习率调度器')
parser.add_argument('--warmup', type=int, default=5, help='学习率的warmup参数')
parser.add_argument('--step_size', type=float, default=100, help='StepLR的step_size')
parser.add_argument('--gamma', type=float, default=0.1, help='StepLR的gamma')
parser.add_argument('--T_max', type=int, default=200, help='CosineAnnealingLR的T_max')
parser.add_argument('--replay_T_max', type=int, default=20, help='重放时CosineAnnealingLR的T_max')
# 训练及重放相关参数 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# 文件路径相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
parser.add_argument('--dataset_path', type=str, default='./dataset', help='数据集的根路径')
parser.add_argument('--root_path', type=str, default='./logs', help='文件保存文件夹的根路径')
# 文件路径相关参数 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# 实验相关设置 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
parser.add_argument('--seed', type=int, default=2024, help='随机种子')
parser.add_argument('--experiment_name', type=str, default="miniimagenet", help='实验名称')
parser.add_argument('--fed_algorithm', type=str, default='SCAFFOLD', help='使用的联邦算法')
parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"], help="实验设备")

parser.add_argument('--use_hlop', action='store_true', help='是否使用hlop')
parser.add_argument('--hlop_start_epochs', type=int, default=0, help='hlop更新的开始轮次')
parser.add_argument("--use_replay", action='store_true', help='是否使用重放')
parser.add_argument('--memory_size', type=int, default=50, help='重放的记忆大小')

parser.add_argument('--device_id', type=str, default="0", help='实验设备的id')
parser.add_argument('--dirichlet', action='store_true', help='使用迪利克雷浓度分配本地数据集')
parser.add_argument('--emd', action='store_true', help='使用EMD距离分配本地数据集')
# 实验相关设置 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

args = parser.parse_args()

# 全局设置随机种子，使得实验结果可以重现
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设置本地训练数据集信息和本地个数
with open("client_dataset_config.yaml", 'r', encoding='utf-8') as f:
    data = yaml.load(f, Loader = yaml.FullLoader)
    if args.dirichlet:
        args.dirichlet_concentration = data['dirichlet_concentration']
        args.n_client = len(args.dirichlet_concentration)
    elif args.emd:
        args.emd_distance = data['emd_distance']
        args.n_client = len(args.emd_distance)

args.root_path = os.path.join('logs', args.experiment_name, args.fed_algorithm + time.strftime(" %Y-%m-%d %H：%M：%S"))
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

if args.device == 'cuda' and not torch.cuda.is_available():
    print("\ncuda is not avaiable.\n")
    args.device = 'cpu'

print('=' * 50)
print('实验名称: {}'.format(args.experiment_name))
print('联邦算法: {}'.format(args.fed_algorithm))
print('使用设备: {}'.format(args.device))
if args.device == "cuda":
    print("cuda设备id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

print('全局通信轮次: {}'.format(args.global_rounds))
print('本地训练轮次: {}'.format(args.local_epochs))
print('使用hlop: {}'.format(args.use_hlop))
print('使用重放: {}'.format(args.use_replay))
if args.use_replay:
    print('本地重放轮次: {}'.format(args.replay_global_rounds))
    print('重放记忆大小：{}'.format(args.memory_size))
print('本地客户端数量： {}'.format(args.n_client))
if args.dirichlet:
    print('本地训练数据集分配方式： 迪利克雷浓度 {}'.format(args.dirichlet_concentration))
elif args.emd:
    print('本地训练数据集分配方式： EMD距离 {}'.format(args.emd_distance))
print('=' * 50)
