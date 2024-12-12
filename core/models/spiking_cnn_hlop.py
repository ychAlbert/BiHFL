import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules.neuron_dsr import LIFNeuron, IFNeuron
from core.modules.neuron_dsr import rate_spikes, weight_rate_spikes
from core.modules.proj_conv import Conv2dProj, SSConv2dProj
from core.modules.proj_linear import LinearProj, SSLinear, SSLinearProj
from core.modules.hlop_module import HLOP

__all__ = ['spiking_cnn']

cfg = {'A': [64, 'M', 128, 'M', 256, 'M']}


class CNN(nn.Module):
    def __init__(self, snn_setting, cnn_name, num_classes=10, share_classifier=False,
                 neuron_type='lif', fc_size=4096, ss=False, hlop_with_wfr=True,
                 hlop_spiking=False, hlop_spiking_scale=20.,
                 hlop_spiking_timesteps=1000., proj_type='input'):
        super(CNN, self).__init__()

        self.timesteps = snn_setting['timesteps']
        self.snn_setting = snn_setting
        self.neuron_type = neuron_type

        self.share_classifier = share_classifier
        self.ss = ss
        self.hlop_with_wfr = hlop_with_wfr
        self.hlop_spiking = hlop_spiking
        self.hlop_spiking_scale = hlop_spiking_scale
        self.hlop_spiking_timesteps = hlop_spiking_timesteps
        # choice for projection type in bottom implementation
        # it is theoretically equivalent for input and weight, while weight enables acceleration of convolutional operations
        self.proj_type = proj_type

        self.init_channels = 3
        self.features, self.hlop_modules = self._make_layers(cfg[cnn_name])

        self.fc_size = fc_size

        if self.neuron_type == 'lif':
            self.tau = snn_setting['tau']
            self.delta_t = snn_setting['delta_t']
            self.weight_avg = True
        elif self.neuron_type == 'if':
            self.weight_avg = False
        else:
            raise NotImplementedError('Please use IF or LIF model.')

        if share_classifier:
            self.hlop_modules.append(HLOP(fc_size, spiking=self.hlop_spiking,
                                          spiking_scale=self.hlop_spiking_scale,
                                          spiking_timesteps=self.hlop_spiking_timesteps))
            if self.ss:
                self.classifiers = nn.ModuleList([SSLinearProj(fc_size, num_classes, bias=False)])
            else:
                self.classifiers = nn.ModuleList([LinearProj(fc_size, num_classes, bias=False)])
        else:
            if self.ss:
                self.classifiers = nn.ModuleList([SSLinear(fc_size, num_classes)])
            else:
                self.classifiers = nn.ModuleList([nn.Linear(fc_size, num_classes)])
        self.n_classifier = 1

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, Conv2dProj):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear) or isinstance(module, LinearProj):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _make_layers(self, cfg: list):
        layers, hlop_modules = [], []
        for x in cfg:
            if x == 'M':    # 如果x元素是'M'，则增加平均池化层
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                if self.ss:
                    layers.append(SSConv2dProj(self.init_channels, x, kernel_size=3, padding=1, bias=False,
                                               proj_type=self.proj_type))
                else:
                    layers.append(Conv2dProj(self.init_channels, x, kernel_size=3, padding=1, bias=False,
                                             proj_type=self.proj_type))

                layers.append(nn.BatchNorm2d(x))
                # 增加神经元
                if self.neuron_type == 'lif':
                    layers.append(LIFNeuron(self.snn_setting))
                elif self.neuron_type == 'if':
                    layers.append(IFNeuron(self.snn_setting))
                else:
                    raise NotImplementedError('Please use IF or LIF model.')

                hlop_modules.append(HLOP(self.init_channels * 3 * 3, spiking=self.hlop_spiking,
                                         spiking_scale=self.hlop_spiking_scale,
                                         spiking_timesteps=self.hlop_spiking_timesteps))
                self.init_channels = x

        return nn.Sequential(*layers), nn.ModuleList(hlop_modules)

    def forward(self, x, task_id=None, projection=False, proj_id_list=[0], update_hlop=False,
                fix_subspace_id_list=None):
        inputs = torch.cat([x[:, i, :, :, :] for i in range(self.timesteps)], 0)

        # 提取特征
        index = 0
        # 对于每个特征提取层
        for feature_extra_layer in self.features:
            # 如果是Conv2dProj/LinearProj类型
            if isinstance(feature_extra_layer, Conv2dProj) or isinstance(feature_extra_layer, LinearProj):
                if projection:
                    proj_func = self.hlop_modules[index].get_proj_func(subspace_id_list=proj_id_list)
                    x_ = feature_extra_layer(inputs, projection=True, proj_func=proj_func)
                else:
                    x_ = feature_extra_layer(inputs, projection=False)
                if update_hlop:
                    if self.hlop_with_wfr:
                        # update hlop by weighted firing rate
                        inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
                    if isinstance(feature_extra_layer, Conv2dProj):
                        inputs = F.unfold(inputs, feature_extra_layer.kernel_size,
                                          dilation=feature_extra_layer.dilation, padding=feature_extra_layer.padding,
                                          stride=feature_extra_layer.stride).transpose(1, 2)
                        inputs = inputs.reshape(-1, inputs.shape[2])
                    with torch.no_grad():
                        self.hlop_modules[index].forward_with_update(inputs, fix_subspace_id_list=fix_subspace_id_list)
                index += 1
                inputs = x_
            # 如果是其余类型类型
            else:
                inputs = feature_extra_layer(inputs)

        out = inputs.view(inputs.size(0), -1)

        out_before_clf = out

        # 如果不共享分类层
        if not self.share_classifier:
            assert task_id is not None
            # 针对任务task_id的分类器进行推导
            out = self.classifiers[task_id](out)
        # 如果共享分类层
        else:
            # 使用第一个分类器
            classifier = self.classifiers[0]
            if projection:
                proj_func = self.hlop_modules[index].get_proj_func(subspace_id_list=proj_id_list)
                out_ = classifier(out, projection=True, proj_func=proj_func)
            else:
                out_ = classifier(out, projection=False)
            if update_hlop:
                if self.hlop_with_wfr:
                    # update hlop by weighted firing rate
                    out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
                with torch.no_grad():
                    self.hlop_modules[index].forward_with_update(out, fix_subspace_id_list=fix_subspace_id_list)
            out = out_

        if self.weight_avg:
            out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
        else:
            out = rate_spikes(out, self.timesteps)
        return out_before_clf, out

    def forward_features(self, x):
        inputs = torch.cat([x[:, _, :, :, :] for _ in range(self.timesteps)], 0)
        index = 0
        feature_list = []
        for m in self.features:
            if isinstance(m, Conv2dProj) or isinstance(m, LinearProj):
                x_ = m(inputs, projection=False)
                if self.hlop_with_wfr:
                    # calculate weighted firing rate
                    inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
                if isinstance(m, Conv2dProj):
                    inputs = F.unfold(inputs, m.kernel_size, dilation=m.dilation, padding=m.padding,
                                      stride=m.stride).transpose(1, 2)
                    inputs = inputs.reshape(-1, inputs.shape[2])
                feature_list.append(inputs.detach().cpu())
                index += 1
                inputs = x_
            else:
                inputs = m(inputs)

        if self.share_classifier:
            inputs = self.pool(inputs)
            if self.hlop_with_wfr:
                # calculate weighted firing rate
                inputs = weight_rate_spikes(inputs, self.timesteps, self.tau, self.delta_t)
            inputs = inputs.view(inputs.size(0), -1)
            feature_list.append(inputs.detach().cpu())

        return feature_list

    def add_classifier(self, num_classes):
        """
        增加分类器
        :param num_classes 分类的类别数
        @return:
        """
        # 分类器数量增加
        self.n_classifier += 1
        # 增加线性层分类器
        if self.ss:
            self.classifiers.append(SSLinear(self.fc_size, num_classes).to(self.classifiers[0].weight.device))
        else:
            self.classifiers.append(nn.Linear(self.fc_size, num_classes).to(self.classifiers[0].weight.device))
        # 将最后一个分类器层的权重初始化为均值为0，标准差为0.01的正态分布并将偏置初始化为0
        last_classifier = self.classifiers[-1]
        last_classifier.weight.data.normal_(0, 0.01)
        if last_classifier.bias is not None:
            last_classifier.bias.data.zero_()

    def merge_hlop_subspace(self):
        """
        合并HLOP模块的子空间
        Returns:

        """
        for module in self.hlop_modules:
            module.merge_subspace()

    def add_hlop_subspace(self, out_features):
        """
        增加HLOP模块的子空间
        Args:
            out_features: 输出特征维度

        Returns:

        """
        if isinstance(out_features, list):  # out_numbers如果是一个list
            for i in range(len(self.hlop_modules)):  # 对于每一个hlop_modules，增加其子空间
                self.hlop_modules[i].add_subspace(out_features[i])
        else:  # out_numbers如果不是list（是int）
            for module in self.hlop_modules:  # 对于每一个hlop_modules，增加其子空间
                module.add_subspace(out_features)

    def fix_bn(self):
        """
        固定BN层和神经元参数
        @return:
        """
        for module in self.modules():  # 对于每一个module
            if isinstance(module, nn.BatchNorm2d):  # module如果是nn.BatchNorm2d类型
                module.eval()  # 模型开启评估模式，此时梯度不进行反向传播
                module.weight.requires_grad = False  # 设置module的权重和偏置都没有梯度
                module.bias.requires_grad = False
            if isinstance(module, LIFNeuron) or isinstance(module, IFNeuron):  # module如果是LIFNeuron/IFNeuron类型
                if self.snn_setting['train_Vth']:  # 如果SNN参数值的train_Vth为True
                    module.Vth.requires_grad = False  # 设置module的Vth没有梯度


def spiking_cnn(snn_setting, **kwargs):
    return CNN(snn_setting, 'A', fc_size=4 * 4 * 256, **kwargs)
