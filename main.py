#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : 程序入口
import os
import sys
import csv
from main_set import *
from core.servers.serveravg import FedAvg
from core.servers.serverdyn import FedDyn
from core.servers.servermoon import MOON
from core.servers.serverprox import FedProx
from core.servers.serverscaffold import SCAFFOLD
from core.servers.serverhifa import HIFA

def run(args):
    # 创建result文件夹
    if not os.path.exists('result'):
        os.makedirs('result')
    
    # 创建CSV文件并重定向输出
    csv_file_path = os.path.join('result', f'{args.name}.csv')
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    sys.stdout = csv_file

    # 任务描述(例如元素(0, 10)：任务0有10个类)
    taskcla = None
    # 训练集
    xtrain, ytrain = {}, {}
    # 测试集
    xtest, ytest = {}, {}
    # 模型
    model = None

    # 获取数据集和模型 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # pmnist/pmnist_bptt/pmnist_ottt 实验
    if args.experiment_name.startswith('pmnist'):
        from core.dataloader import pmnist as pmd
        data, taskcla, inputsize = pmd.get(data_dir=args.dataset_path, seed=args.seed)
        for task_id, n_task_class in taskcla:
            xtrain[task_id], ytrain[task_id] = data[task_id]['train']['x'], data[task_id]['train']['y']
            xtest[task_id], ytest[task_id] = data[task_id]['test']['x'], data[task_id]['test']['y']
        n_class = taskcla[0][1]
        # pminst 实验
        if args.experiment_name == 'pmnist':
            from core.models import spiking_MLP
            snn_setting = {}
            snn_setting['timesteps'] = args.timesteps
            snn_setting['train_Vth'] = True if args.train_Vth == 1 else False
            snn_setting['Vth'] = args.Vth
            snn_setting['tau'] = args.tau
            snn_setting['delta_t'] = args.delta_t
            snn_setting['alpha'] = args.alpha
            snn_setting['Vth_bound'] = args.Vth_bound
            snn_setting['rate_stat'] = True if args.rate_stat == 1 else False
            hlop_with_wfr = True
            if args.not_hlop_with_wfr:
                hlop_with_wfr = False
            model = spiking_MLP(snn_setting, num_classes=n_class, n_hidden=800,
                                ss=args.sign_symmetric, fa=args.feedback_alignment, hlop_with_wfr=hlop_with_wfr,
                                hlop_spiking=args.hlop_spiking, hlop_spiking_scale=args.hlop_spiking_scale,
                                hlop_spiking_timesteps=args.hlop_spiking_timesteps)
        # pmnist_bptt 实验
        elif args.experiment_name == 'pmnist_bptt':
            from core.models import spiking_MLP_bptt
            model = spiking_MLP_bptt(num_classes=n_class, n_hidden=800, ss=args.sign_symmetric,
                                     fa=args.feedback_alignment, timesteps=args.timesteps,
                                     hlop_spiking=args.hlop_spiking, hlop_spiking_scale=args.hlop_spiking_scale,
                                     hlop_spiking_timesteps=args.hlop_spiking_timesteps)
        # pmnist_ottt 实验
        elif args.experiment_name == 'pmnist_ottt':
            from core.models import spiking_MLP_ottt
            model = spiking_MLP_ottt(num_classes=n_class, n_hidden=800, ss=args.sign_symmetric,
                                     fa=args.feedback_alignment, timesteps=args.timesteps,
                                     hlop_spiking=args.hlop_spiking,
                                     hlop_spiking_scale=args.hlop_spiking_scale,
                                     hlop_spiking_timesteps=args.hlop_spiking_timesteps)
    # cifar 实验
    elif args.experiment_name == 'cifar':
        from core.dataloader import cifar100 as cf100
        data, taskcla, inputsize = cf100.get(data_dir=args.dataset_path, seed=args.seed)
        for task_id, n_task_class in taskcla:
            xtrain[task_id], ytrain[task_id] = data[task_id]['train']['x'], data[task_id]['train']['y']
            xtest[task_id], ytest[task_id] = data[task_id]['test']['x'], data[task_id]['test']['y']
        n_class = taskcla[0][1]
        from core.models import spiking_cnn
        snn_setting = {}
        snn_setting['timesteps'] = args.timesteps
        snn_setting['train_Vth'] = True if args.train_Vth == 1 else False
        snn_setting['Vth'] = args.Vth
        snn_setting['tau'] = args.tau
        snn_setting['delta_t'] = args.delta_t
        snn_setting['alpha'] = args.alpha
        snn_setting['Vth_bound'] = args.Vth_bound
        snn_setting['rate_stat'] = True if args.rate_stat == 1 else False
        hlop_with_wfr = True
        if args.not_hlop_with_wfr:
            hlop_with_wfr = False
        model = spiking_cnn(snn_setting, num_classes=n_class, ss=args.sign_symmetric, hlop_with_wfr=hlop_with_wfr,
                            hlop_spiking=args.hlop_spiking, hlop_spiking_scale=args.hlop_spiking_scale,
                            hlop_spiking_timesteps=args.hlop_spiking_timesteps, proj_type=args.hlop_proj_type)
    # miniimagenet 实验
    elif args.experiment_name == 'miniimagenet':
        from core.dataloader import miniimagenet as data_loader
        dataloader = data_loader.DatasetGen(data_dir=args.dataset_path, seed=args.seed)
        taskcla, inputsize = dataloader.taskcla, dataloader.inputsize
        for task_id, n_task_class in taskcla:
            data = dataloader.get(task_id)
            xtrain[task_id], ytrain[task_id] = data[task_id]['train']['x'], data[task_id]['train']['y']
            xtest[task_id], ytest[task_id] = data[task_id]['test']['x'], data[task_id]['test']['y']
        n_class = taskcla[0][1]
        from core.models import spiking_resnet18
        snn_setting = {}
        snn_setting['timesteps'] = args.timesteps
        snn_setting['train_Vth'] = True if args.train_Vth == 1 else False
        snn_setting['Vth'] = args.Vth
        snn_setting['tau'] = args.tau
        snn_setting['delta_t'] = args.delta_t
        snn_setting['alpha'] = args.alpha
        snn_setting['Vth_bound'] = args.Vth_bound
        snn_setting['rate_stat'] = True if args.rate_stat == 1 else False
        
        hlop_with_wfr = True
        if args.not_hlop_with_wfr:
            hlop_with_wfr = False
        
        model = spiking_resnet18(snn_setting, num_classes=n_class, nf=20, ss=args.sign_symmetric, 
                                 hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking, 
                                 hlop_spiking_scale=args.hlop_spiking_scale, 
                                 hlop_spiking_timesteps=args.hlop_spiking_timesteps, proj_type=args.hlop_proj_type)
    # fivedataset/fivedataset_domain 实验
    elif args.experiment_name.startswith('fivedataset'):
        from core.dataloader import five_datasets as data_loader
        data, taskcla, inputsize = data_loader.get(data_dir=args.dataset_path, seed=args.seed)
        for task_id, n_task_class in taskcla:
            xtrain[task_id], ytrain[task_id] = data[task_id]['train']['x'], data[task_id]['train']['y']
            xtest[task_id], ytest[task_id] = data[task_id]['test']['x'], data[task_id]['test']['y']
        n_class = taskcla[0][1]
        from core.models import spiking_resnet18
        snn_setting = {}
        snn_setting['timesteps'] = args.timesteps
        snn_setting['train_Vth'] = True if args.train_Vth == 1 else False
        snn_setting['Vth'] = args.Vth
        snn_setting['tau'] = args.tau
        snn_setting['delta_t'] = args.delta_t
        snn_setting['alpha'] = args.alpha
        snn_setting['Vth_bound'] = args.Vth_bound
        snn_setting['rate_stat'] = True if args.rate_stat == 1 else False
        hlop_with_wfr = True
        if args.not_hlop_with_wfr:
            hlop_with_wfr = False
        if args.experiment_name == 'fivedataset':
            model = spiking_resnet18(snn_setting, num_classes=n_class, nf=20, ss=args.sign_symmetric,
                                     hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking,
                                     hlop_spiking_scale=args.hlop_spiking_scale,
                                     hlop_spiking_timesteps=args.hlop_spiking_timesteps,
                                     proj_type=args.hlop_proj_type)
        elif args.experiment_name == 'fivedataset_domain':
            model = spiking_resnet18(snn_setting, num_classes=n_class, nf=20, ss=args.sign_symmetric,
                                     hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking,
                                     hlop_spiking_scale=args.hlop_spiking_scale,
                                     hlop_spiking_timesteps=args.hlop_spiking_timesteps,
                                     proj_type=args.hlop_proj_type, share_classifier=True)

    # 获取联邦算法 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if args.fed_algorithm.lower() == 'fedavg':
        server = FedAvg(args, xtrain, ytrain, xtest, ytest, taskcla, model)
    elif args.fed_algorithm.lower() == 'scaffold':
        server = SCAFFOLD(args, xtrain, ytrain, xtest, ytest, taskcla, model)
    elif args.fed_algorithm.lower() == 'fedprox':
        server = FedProx(args, xtrain, ytrain, xtest, ytest, taskcla, model)
    elif args.fed_algorithm.lower() == 'feddyn':
        server = FedDyn(args, xtrain, ytrain, xtest, ytest, taskcla, model)
    elif args.fed_algorithm.lower() == 'moon':
        server = MOON(args, xtrain, ytrain, xtest, ytest, taskcla, model)
    elif args.fed_algorithm.lower() == 'hifa':
        server = HIFA(args, xtrain, ytrain, xtest, ytest, taskcla, model)

    server.execute()

    # 关闭CSV文件
    csv_file.close()


if __name__ == "__main__":
    run(args)
