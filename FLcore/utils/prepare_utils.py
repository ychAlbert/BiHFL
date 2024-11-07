#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : 实验准备的相关工具（具体功能为：根据进行的实验名称，给出对应的实验参数）
def prepare_bptt_ottt(experiment_name: str):
    """
    根据实验名称准备bptt和ottt的值
    @param experiment_name: 实验名称
    @return:
    """
    bptt, ottt = False, False
    if experiment_name.endswith('bptt'):
        bptt, ottt = True, False
    elif experiment_name.endswith('ottt'):
        bptt, ottt = False, True
    return bptt, ottt


def prepare_hlop_out(experiment_name: str):
    """
    根据实验名称准备hlop_out_XXX相关的值
    @param experiment_name:
    @return:
    """
    hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1 = [], [], []
    if experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
        hlop_out_num = [80, 200, 100]
        hlop_out_num_inc = [70, 70, 70]
    elif experiment_name == 'cifar':  # cifar 实验
        hlop_out_num = [6, 100, 200]
        hlop_out_num_inc = [2, 20, 40]
    elif experiment_name == 'miniimagenet':  # miniimagenet 实验
        hlop_out_num = [24, [90, 90], [90, 90], [90, 180, 10], [180, 180], [180, 360, 20], [360, 360], [360, 720, 40],
                        [720, 720]]
        hlop_out_num_inc = [2, [6, 6], [6, 6], [6, 12, 1], [12, 12], [12, 24, 2], [24, 24], [24, 48, 4], [48, 48]]
        hlop_out_num_inc1 = [0, [2, 2], [2, 2], [2, 4, 0], [4, 4], [4, 8, 0], [8, 8], [8, 16, 0], [16, 16]]
    elif experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
        hlop_out_num = [6, [40, 40], [40, 40], [40, 100, 6], [100, 100], [100, 200, 8], [200, 200], [200, 200, 16],
                        [200, 200]]
        hlop_out_num_inc = [6, [40, 40], [40, 40], [40, 100, 6], [100, 100], [100, 200, 8], [200, 200], [200, 200, 16],
                            [200, 200]]
    return hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1


def prepare_dataset(experiment_name: str, dataset_path: str, seed: int):
    """
    根据实验名称准备hlop_out_XXX相关的值
    @param experiment_name:
    @param dataset_path:
    @param seed:
    @return:
    """
    xtrain, ytrain, xtest, ytest, taskcla = {}, {}, {}, {}, None
    if experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
        from ..dataloader import pmnist as pmd
        data, taskcla, inputsize = pmd.get(data_dir=dataset_path, seed=seed)
        for task_id, ncla in taskcla:
            xtrain[task_id] = data[task_id]['train']['x']
            ytrain[task_id] = data[task_id]['train']['y']
            xtest[task_id] = data[task_id]['test']['x']
            ytest[task_id] = data[task_id]['test']['y']
    elif experiment_name == 'cifar':  # cifar 实验
        from ..dataloader import cifar100 as cf100
        data, taskcla, inputsize = cf100.get(data_dir=dataset_path, seed=seed)
        for task_id, ncla in taskcla:
            xtrain[task_id] = data[task_id]['train']['x']
            ytrain[task_id] = data[task_id]['train']['y']
            xtest[task_id] = data[task_id]['test']['x']
            ytest[task_id] = data[task_id]['test']['y']
    elif experiment_name == 'miniimagenet':  # miniimagenet 实验
        from ..dataloader import miniimagenet as data_loader
        dataloader = data_loader.DatasetGen(data_dir=dataset_path, seed=seed)
        taskcla, inputsize = dataloader.taskcla, dataloader.inputsize
        for task_id, ncla in taskcla:
            data = dataloader.get(task_id)
            xtrain[task_id] = data[task_id]['train']['x']
            ytrain[task_id] = data[task_id]['train']['y']
            xtest[task_id] = data[task_id]['test']['x']
            ytest[task_id] = data[task_id]['test']['y']
    elif experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
        from ..dataloader import five_datasets as data_loader
        data, taskcla, inputsize = data_loader.get(data_dir=dataset_path, seed=seed)
        for task_id, ncla in taskcla:
            xtrain[task_id] = data[task_id]['train']['x']
            ytrain[task_id] = data[task_id]['train']['y']
            xtest[task_id] = data[task_id]['test']['x']
            ytest[task_id] = data[task_id]['test']['y']
    return xtrain, ytrain, xtest, ytest, taskcla


def prepare_model(experiment_name: str, args, ncla):
    model = None
    if experiment_name == 'pmnist':  # pmnist 实验
        from ..models import spiking_MLP
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
        model = spiking_MLP(snn_setting, num_classes=ncla, n_hidden=800,
                            ss=args.sign_symmetric,
                            fa=args.feedback_alignment, hlop_with_wfr=hlop_with_wfr,
                            hlop_spiking=args.hlop_spiking,
                            hlop_spiking_scale=args.hlop_spiking_scale,
                            hlop_spiking_timesteps=args.hlop_spiking_timesteps)
    elif experiment_name == 'pmnist_bptt':  # pmnist_bptt 实验
        from ..models import spiking_MLP_bptt
        model = spiking_MLP_bptt(num_classes=ncla, n_hidden=800, ss=args.sign_symmetric,
                         fa=args.feedback_alignment, timesteps=args.timesteps,
                         hlop_spiking=args.hlop_spiking, hlop_spiking_scale=args.hlop_spiking_scale,
                         hlop_spiking_timesteps=args.hlop_spiking_timesteps)
    elif experiment_name == 'pmnist_ottt':  # pmnist_ottt 实验
        from ..models import spiking_MLP_ottt
        model = spiking_MLP_ottt(num_classes=ncla, n_hidden=800, ss=args.sign_symmetric, fa=args.feedback_alignment,
                         timesteps=args.timesteps, hlop_spiking=args.hlop_spiking,
                         hlop_spiking_scale=args.hlop_spiking_scale, hlop_spiking_timesteps=args.hlop_spiking_timesteps)
    elif experiment_name == 'cifar':  # cifar 实验
        from ..models import spiking_cnn
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
        model = spiking_cnn(snn_setting, num_classes=ncla, ss=args.sign_symmetric,
                            hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking,
                            hlop_spiking_scale=args.hlop_spiking_scale,
                            hlop_spiking_timesteps=args.hlop_spiking_timesteps,
                            proj_type=args.hlop_proj_type)
    elif experiment_name == 'miniimagenet':  # miniimagenet 实验
        from ..models import spiking_resnet18
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
        model = spiking_resnet18(snn_setting, num_classes=ncla, nf=20, ss=args.sign_symmetric,
                                 hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking,
                                 hlop_spiking_scale=args.hlop_spiking_scale,
                                 hlop_spiking_timesteps=args.hlop_spiking_timesteps,
                                 proj_type=args.hlop_proj_type, first_conv_stride2=True)
    elif experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
        from ..models import spiking_resnet18
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
        if experiment_name == 'fivedataset':
            model = spiking_resnet18(snn_setting, num_classes=ncla, nf=20, ss=args.sign_symmetric,
                                     hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking,
                                     hlop_spiking_scale=args.hlop_spiking_scale,
                                     hlop_spiking_timesteps=args.hlop_spiking_timesteps,
                                     proj_type=args.hlop_proj_type)
        elif experiment_name == 'fivedataset_domain':
            model = spiking_resnet18(snn_setting, num_classes=ncla, nf=20, ss=args.sign_symmetric,
                                     hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking,
                                     hlop_spiking_scale=args.hlop_spiking_scale,
                                     hlop_spiking_timesteps=args.hlop_spiking_timesteps,
                                     proj_type=args.hlop_proj_type, share_classifier=True)
    return model