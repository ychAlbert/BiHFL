import os

import numpy as np
import torch
from torchvision import datasets, transforms


def get(data_dir='./data/', seed=0):
    data = {}
    taskcla = []
    size = [3, 32, 32]

    file_dir = os.path.join(data_dir, 'binary_cifar100')
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        cifar100_tensor_data = {}
        cifar100_tensor_data['train'] = datasets.CIFAR100(data_dir, train=True, download=True,
                                                         transform=transforms.Compose(
                                                             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        cifar100_tensor_data['test'] = datasets.CIFAR100(data_dir, train=False, download=True,
                                                        transform=transforms.Compose(
                                                            [transforms.ToTensor(), transforms.Normalize(mean, std)]))

        for n in range(10):
            data[n] = {}
            data[n]['name'] = 'cifar100'
            data[n]['ncla'] = 10
            data[n]['train'] = {'x': [], 'y': []}
            data[n]['test'] = {'x': [], 'y': []}

        for type_ in ['train', 'test']:
            loader = torch.utils.data.DataLoader(cifar100_tensor_data[type_], batch_size=1, shuffle=False)
            for image, target in loader:
                n = target.numpy()[0]
                nn = (n // 10)
                data[nn][type_]['x'].append(image)  # 255
                data[nn][type_]['y'].append(n % 10)

        # "Unify" and save
        for key_ in data.keys():
            for type_ in ['train', 'test']:
                data[key_][type_]['x'] = torch.stack(data[key_][type_]['x']).view(-1, size[0], size[1], size[2])
                data[key_][type_]['y'] = torch.LongTensor(np.array(data[key_][type_]['y'], dtype=int)).view(-1)
                torch.save(data[key_][type_]['x'], os.path.join(os.path.expanduser(file_dir), 'data' + str(key_) + type_ + 'x.bin'))
                torch.save(data[key_][type_]['y'], os.path.join(os.path.expanduser(file_dir), 'data' + str(key_) + type_ + 'y.bin'))

    # Load binary files
    data = {}
    # ids=list(shuffle(np.arange(5),random_state=seed))
    ids = list(np.arange(10))
    print('Task order =', ids)
    for i in range(10):
        data[i] = dict.fromkeys(['name', 'ncla', 'train', 'test'])
        for type_ in ['train', 'test']:
            data[i][type_] = {'x': [], 'y': []}
            data[i][type_]['x'] = torch.load(os.path.join(os.path.expanduser(file_dir), 'data' + str(ids[i]) + type_ + 'x.bin'))
            data[i][type_]['y'] = torch.load(os.path.join(os.path.expanduser(file_dir), 'data' + str(ids[i]) + type_ + 'y.bin'))
        data[i]['ncla'] = len(np.unique(data[i]['train']['y'].numpy()))
        if data[i]['ncla'] == 2:
            data[i]['name'] = 'cifar10-' + str(ids[i])
        else:
            data[i]['name'] = 'cifar100-' + str(ids[i])

    # Validation
    # for t in data.keys():
    #    r=np.arange(data[t]['train']['x'].size(0))
    #    r=np.array(shuffle(r,random_state=seed),dtype=int)
    #    nvalid=int(pc_valid*len(r))
    #    ivalid=torch.LongTensor(r[:nvalid])
    #    itrain=torch.LongTensor(r[nvalid:])
    #    data[t]['valid']={}
    #    data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
    #    data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
    #    data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
    #    data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for key_ in data.keys():
        taskcla.append((key_, data[key_]['ncla']))
        n += data[key_]['ncla']
    data['ncla'] = n

    return data, taskcla, size
