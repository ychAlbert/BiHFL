# Permuted MNIST
import os
import sys

import numpy as np
import torch
from sklearn.utils import shuffle
from torchvision import datasets, transforms


def get(data_dir='./dataset/', seed=0, fixed_order=False):
    data = {}  # 存放每个task的数据
    taskcla = []  #
    size = [1, 28, 28]  # 图像输入size

    mnist_dir = data_dir
    pmnist_dir = os.path.join(mnist_dir, 'binary_pmnist')

    nperm = 10  # 10个task
    seeds = np.array(list(range(nperm)), dtype=int)  # 生成种子
    if not fixed_order:  # 如果次序是不固定的
        seeds = shuffle(seeds, random_state=seed)  # 打乱seed

    if not os.path.isdir(pmnist_dir):  # 如果不存在数据集的文件夹
        os.makedirs(pmnist_dir)  # 创建数据集的文件夹
        mean = (0.1307,)  # MNIST数据集的均值
        std = (0.3081,)  # MNIST数据集的标准差
        dat = {}
        dat['train'] = datasets.MNIST(mnist_dir, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        dat['test'] = datasets.MNIST(mnist_dir, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))

        for i, r in enumerate(seeds):  # 对于索引i处的值r（类别）
            print(i, end=',')
            sys.stdout.flush()
            data[i] = {}
            data[i]['name'] = 'pmnist-{:d}'.format(i)  # data中键i的name键的值
            data[i]['ncla'] = 10  # data中键i的ncla键的值
            for s in ['train', 'test']:  # 对于训练集和测试机
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[i][s] = {'x': [], 'y': []}  # data中键i的数据集类别键的x（图像）键和y（标签）键
                for image, target in loader:
                    aux = image.view(-1).numpy()
                    aux = shuffle(aux, random_state=r * 100 + i)  # 对MNIST中所有图像以相同的方式进行像素重排
                    image = torch.FloatTensor(aux).view(size)
                    data[i][s]['x'].append(image)
                    data[i][s]['y'].append(target.numpy()[0])

            for s in ['train', 'test']:
                # 调整data中键i的数据集类别键的x（图像）键和y（标签）键的格式
                data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
                data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)
                # 保存
                torch.save(data[i][s]['x'], os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                torch.save(data[i][s]['y'], os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))
    else:
        for i, r in enumerate(seeds):   # 对于索引i处的值r（类别）
            data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
            data[i]['ncla'] = 10        # data中键i的ncla键的值
            data[i]['name'] = 'pmnist-{:d}'.format(i)   # data中键i的name键的值

            for s in ['train', 'test']:
                data[i][s] = {'x': [], 'y': []}
                data[i][s]['x'] = torch.load(
                    os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(
                    os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(r) + s + 'y.bin'))

    # Validation
    # for t in data.keys():
    #    r=np.arange(data[t]['train']['x'].size(0))
    #    # r=np.array(shuffle(r,random_state=seed),dtype=int)
    #    r=np.array(r,dtype=int)
    #    nvalid=int(pc_valid*len(r))
    #    ivalid=torch.LongTensor(r[:nvalid])
    #    itrain=torch.LongTensor(r[nvalid:])
    #    data[t]['valid'] = {}
    #    data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
    #    data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
    #    data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
    #    data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for t in data.keys():   # 对于data中每一个键（实则上是对于每一个task）
        taskcla.append((t, data[t]['ncla']))    # taskcla为task的id和该id对应task的ncla
        n += data[t]['ncla']    # data中键t的ncla键（猜测ncla为num_classify，即分类的类别数）
    data['ncla'] = n    # 一共的task的分类的类别数
    return data, taskcla, size

########################################################################################################################
