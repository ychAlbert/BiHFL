# 获取命令行参数解析对象
import argparse
import os
import time

import torch

from FLcore.servers.serveravg import FedAvg
from FLcore.servers.serverdyn import FedDyn
from FLcore.servers.servermoon import MOON
from FLcore.servers.serverprox import FedProx
from FLcore.servers.serverscaffold import SCAFFOLD
from FLcore.utils import prepare_dataset, prepare_model


def run(args):
    xtrain, ytrain, xtest, ytest, taskcla = prepare_dataset(args.experiment_name, args.dataset_path, 0)
    model = prepare_model(args.experiment_name, args, 10)
    model = model.to(args.device)

    if args.fed_algorithm == 'FedAvg':
        server = FedAvg(args, xtrain, ytrain, xtest, ytest, taskcla, model)
    elif args.fed_algorithm == 'SCAFFOLD':
        server = SCAFFOLD(args, xtrain, ytrain, xtest, ytest, taskcla, model)
    elif args.fed_algorithm == 'FedProx':
        server = FedProx(args, xtrain, ytrain, xtest, ytest, taskcla, model)
    elif args.fed_algorithm == 'FedDyn':
        server = FedDyn(args, xtrain, ytrain, xtest, ytest, taskcla, model)
    elif args.fed_algorithm == 'MOON':
        server = MOON(args, xtrain, ytrain, xtest, ytest, taskcla, model)

    server.execute()


if __name__ == "__main__":
    _seed_ = 2024
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()


    # SNN settings
    parser.add_argument('-timesteps', default=20, type=int)
    parser.add_argument('-Vth', default=0.3, type=float)
    parser.add_argument('-tau', default=1.0, type=float)
    parser.add_argument('-delta_t', default=0.05, type=float)
    parser.add_argument('-alpha', default=0.3, type=float)
    parser.add_argument('-train_Vth', default=1, type=int)
    parser.add_argument('-Vth_bound', default=0.0005, type=float)
    parser.add_argument('-rate_stat', default=0, type=int)

    parser.add_argument('-not_hlop_with_wfr', action='store_true', help='use spikes for hlop update')
    parser.add_argument('-hlop_spiking', action='store_true', help='use hlop with lateral spiking neurons')
    parser.add_argument('-hlop_spiking_scale', default=20., type=float)
    parser.add_argument('-hlop_spiking_timesteps', default=1000., type=float)

    # 普遍参数
    parser.add_argument('-go', "--goal", type=str, default="test", help="The goal for this experiment")

    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")

    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)

    # 实际参数？
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0, help="参与训练但中途退出的客户端比例")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0, help="本地训练时，速度慢的客户端比例")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0, help="发送全局模型时，速度慢的客户端比例")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="是否根据时间成本对每轮客户进行分组和选择")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000, help="丢弃慢客户端的阈值")

    # 联邦算法相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser.add_argument("--FedDyn_alpha", type=float, default=1.0, help="FedDyn算法的α参数")  # FedDyn 相关参数
    parser.add_argument("--FedProx_mu", type=float, default=0.1, help="FedProx算法的μ参数")  # FedProx 相关参数
    parser.add_argument("--MOON_tau", type=float, default=1.0, help="MOON算法的τ参数")  # MOON 相关参数
    parser.add_argument("--MOON_mu", type=float, default=1.0, help="MOON算法的μ参数")
    parser.add_argument("--SCAFFOLD_eta", type=float, default=1.0, help="SCAFFOLD算法的η参数")  # SCAFFOLD 相关参数
    # 联邦算法相关参数 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # 文件路径相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser.add_argument("--dataset_path", type=str, default='./dataset', help="数据集的根路径")
    parser.add_argument("--root_path", type=str, default='./logs', help="文件保存文件夹的根路径")
    # 文件路径相关参数 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # 训练及重放相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser.add_argument("--global_rounds", type=int, default=1, help="全局通信轮次")
    parser.add_argument("--local_epochs", type=int, default=1, help="本地训练轮次")
    parser.add_argument("--batch_size", type=int, default=16, help="训练数据批处理大小")
    parser.add_argument("--replay_global_rounds", type=int, default=1, help="重放全局通信轮次")
    parser.add_argument("--replay_local_epochs", type=int, default=1, help="重放本地回放轮次")
    parser.add_argument("--replay_batch_size", type=int, default=64, help="重放数据批处理大小")

    parser.add_argument('--opt', type=str, help='use which optimizer. SGD or Adam', default='SGD')
    parser.add_argument('--momentum', default=0.9, type=float, help="SGD的冲量")
    parser.add_argument('--learning_rate', default=0.01, type=float, help="客户端学习率")
    parser.add_argument('--continual_learning_rate', default=0.01, type=float, help="持续任务的学习率")
    parser.add_argument('--replay_learning_rate', default=0.001, type=float, help="客户端重放学习率")

    parser.add_argument('--lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('--warmup', default=5, type=int, help='warmup epochs for learning rate')
    parser.add_argument('--step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('--T_max', default=200, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('--replay_T_max', default=20, type=int, help='T_max for CosineAnnealingLR for replay')

    # ---------------------------------------------------------------------------------------------------------------- #

    parser.add_argument('--memory_size', default=50, type=int, help='memory size for replay')
    parser.add_argument('--feedback_alignment', action='store_true', help='feedback alignment')
    parser.add_argument('--sign_symmetric', action='store_true', help='use sign symmetric')
    # 训练及重放相关参数 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    parser.add_argument("--experiment_name", type=str, default="pmnist", help="实验名称")
    parser.add_argument('--fed_algorithm', type=str, default='FedAvg', help='使用的联邦算法')

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="实验设备")
    parser.add_argument("--device_id", type=str, default="0", help="实验设备的id")
    parser.add_argument("--num_clients", type=int, default=3, help="客户端数量")

    parser.add_argument('--use_hlop', action='store_true', help="是否使用hlop")
    parser.add_argument('--hlop_start_epochs', default=0, type=int, help='the start epoch to update hlop')
    parser.add_argument('--hlop_proj_type', type=str,
                        help='choice for projection type in bottom implementation, default is input, can choose weight for acceleration of convolutional operations',
                        default='input')

    parser.add_argument("--use_replay", action='store_true', help="是否使用重放")

    # 解析命令行参数
    args = parser.parse_args()

    args.root_path = os.path.join('logs',
                                  args.experiment_name,
                                  args.fed_algorithm + time.strftime(" %Y-%m-%d %H：%M：%S"))
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    print("Algorithm: {}".format(args.fed_algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Using device: {}".format(args.device))

    print("Global rounds: {}".format(args.global_rounds))
    # if args.device == "cuda":
    #     print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print('Using HLOP: {}'.format(args.use_hlop))
    print('Using replay: {}'.format(args.use_replay))
    print("=" * 50)

    run(args)
