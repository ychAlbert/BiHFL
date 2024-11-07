import torch
from torch import nn


def clone_model(model, target):
    """
    将原始模型（model）的模型参数进行克隆并获得目标模型（target）
    :param model: 被克隆的原始模型
    :param target: 克隆出的目标模型
    :return:
    """
    for param, target_param in zip(model.parameters(), target.parameters()):
        target_param.data = param.data.clone()
        # target_param.grad = param.grad.clone()


def update_parameters(model, new_params):
    """
    利用新的参数（new_params）更新模型（model）的参数
    :param model: 参数待更新的模型
    :param new_params: 新的参数
    :return:
    """
    for param, new_param in zip(model.parameters(), new_params):
        param.data = new_param.data.clone()


def reset_net(net: nn.Module):
    """
    将网络的状态重置。做法是遍历网络中的所有 ``Module``，若含有 ``reset()`` 函数，则调用。
    @param net: 任何属于 ``nn.Module`` 子类的网络
    @return:
    """
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()


def model_parameter_vector(model):
    """
    将给定模型model的所有参数整合成一个一维的向量
    @param model: 给定模型
    @return: 整合后的一维向量
    """
    param = [param.view(-1) for param in model.parameters()]
    return torch.cat(param, dim=0)
